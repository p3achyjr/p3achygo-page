---
layout: page
title: Methods
permalink: /engineering/
---

These are some engineering tidbits that I thought were interesting while developing p3achygo. Hopefully there's something in here that's fun :)

## Board Implementation

Implementing a Go Board is like a shrimp gumbo of little algorithms and data structures. There's tons of material on how to write a good board representation, as chess engine folks have been working on these problems for a long time. Here are some examples of tidbits I've implemented:

### Zobrist Hashing

Most AlphaZero reproductions in Go use positional superko rules by default. All this means is that no board position can be repeated. Thus, before we can play any move, we need to check whether the resulting position has occurred yet in the game history.

Rather than storing all previous positions and searching for a match every time, zobrist hashing is a way to compute a hash of the current board position in an online fashion. The way it works is this:

- For an n x n board with m pieces per board intersection, generate n * n * (m + 1) random bitstrings. The (+ 1) term is to account for empty intersections.
- For each board coordinate (i, j) and piece p, map the tuple (i, j, p) to a unique bitstring.
- At game start, calculate the board's initial hash by xor'ing all empty bitstring values together. So essentially xor the bistrings at (i, j, EMPTY) for all (i, j). We can call this value _H_.
- For each move, figure out the transitions on the board the move incurs. In Go, this will flip one intersection from EMPTY to (BLACK | WHITE), and some number of intersections from (BLACK | WHITE) to EMPTY for captured stones.
- For each transition (i, j, p0, p1), xor the bitstrings at (i, j, p0) and (i, j, p1) with _H_. Since xor'ing any value with itself is 0 and xor is associative/commutative, the first xor is essentially "undoing" the last transition in this intersection. The second xor then "commits" the current piece to the intersection.

And that's it! After we compute zobrist hashes like this, all we need to do is shove our hashes into a hash table. In my case, I generate bitstrings of length 128. I don't think I've seen any hash collisions yet :)

It's probably actually sufficient to just leave all hashes for EMPTY as 0, and only generate (n * n * m) bitstrings. The effect of this would be that instead of having a hash that represents, for example, (0, 0) -> EMPTY, (0, 1) -> BLACK, (0, 2), EMPTY ..., we would have a hash that represents only the occupied intersections, so (0, 1) -> BLACK, (0, 3) -> WHITE, etc. I haven't done this in p3achygo, but maybe in the future.

### Group and Liberty Tracking

In Go, when a strongly-connected group is fully surrounded, it is captured and removed from the board. A strongly-connected group is one where all stones are connected via up, down, left, or right directions (so no diagonals). In Go, we say that this group has no more "liberties", where a liberty is an empty adjacent intersection.

A naive way to check for captures would be for every move, check all adjacent intersections of the move, find the strongly connected regions at those intersections, and check whether they have any liberties. However, this is linear in the size of each group every time a stone is placed.

To mitigate this, we can introduce a data structure that maps groups to group IDs, and group IDs to various metadata fields, including number of liberties. Then, every time a stone is placed, we simply subtract the liberties for the group IDs at all adjacent opposite-color groups by one. When a group ID reaches zero liberties, we remove the whole group from the board.

By introducing this data structure, we need to make sure that we handle group merging correctly when two groups combine. This is a perfect use case for a union-find data structure, but in my case, I just implemented a simple O(n) algorithm to merge these groups together stone by stone.

In terms of tracking group IDs--I originally used a monotonically increasing group ID counter, but I eventually realized that I can use the coordinate of the first stone of a group as its ID. This saved me a remarkable amount of CPU computation time.

## Self-Play Engine

The most important part of the AlphaZero pipeline is being able to generate examples as quickly as you can. I don't think I did anything super special here, but I can talk about some of the components of the engine, and some of the decisions that I made.

### Concurrent Self-Play Thread Pool

To enable maximum throughput, I implemented a concurrent self-play threads pool. Essentially, we have _n_ threads in parallel, each player a game and using MCTS. All threads share a common neural network input buffer, and load their inputs independently into an assigned slot in the buffer. When all threads have finished loading their data, or after a certain timeout, we send the buffer to the neural network, and threads copy their results back to a thread-local buffer.

The most fun part about this was writing it in a way where minimal locking occurs. Unfortunately, trying to minimize locking also increased the number of race conditions I've had to deal with. Introducing the timeout also introduced some tricky race conditions, one of which I found just a few weeks ago at the time of writing this. Comes with the territory I suppose :)

### MCTS Implementation

Implementing the MCTS algorithm was a ton of fun. I don't think I ran into anything too crazy while writing it, but there are a couple of interesting implementation details:

- Vectorized Softmax. For Gumbel MCTS, each time we search a non-root node, we construct a new policy using the output of a function $$\sigma(p_a, q_a)$$ as logits for the policy. To construct this policy, we need to compute a softmax. Given how often we call this softmax, it made sense to write a vectorized [version](https://github.com/p3achyjr/p3achygo/blob/34fc362a8955aceba6466971b5efba65b54e0bfd/cc/core/vmath.h#L102-L151). Doing so sped up softmax by about half. I am still using the glibc `exp` function, but in the future I could write a vectorized `exp` function as well using some kind of approximation algorithm.
- Limited value update at root. In my initial implementation, I was calculating the value of the root node as the sum of the values of all visits of the node's children. However, this includes the values under nodes that are sampled, and not necessarily nodes that the net would choose to play by itself. In the pathological case, in a board position where a big group is in atari, there is only one plausible move. However, we are still forced to play out $$k$$ moves because of root sampling. Obviously, the value estimates under these moves should not be included, as otherwise our root value estimate would be much lower than it actually is.
