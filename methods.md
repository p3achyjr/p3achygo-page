---
layout: page
permalink: /methods/
katex: True
---

## Methods

This is a survey of methods implemented in p3achygo. I'm assuming a bit of background in the problem space, so I apologize if there are parts that are confusing. Also for brevity, I will use AlphaZero to mean AlphaGo Zero even though they are different.

## Gumbel MCTS

This is a new MCTS algorithm that works well even for very low visit counts. This enables independent developers such as myself to build a meaningful AlphaZero reproduction without leveraging absurd hardware.

In the original AlphaZero, and in open source reproductions, the search algorithm has used some variant of the PUCT algorithm to determine which node to select at each non-leaf node. The PUCT formula is as follows:

$$U(s, a) = c_{puct}P(s, a)\frac{\sum_bN(s, b)}{1 + N(s, a)}$$
$$a_{\text{next}} = max_a(Q(s, a) + U(s, a))$$

where $s$ is a given game state, $a$ is a given action, $P(s, a)$ is a prior mass for selecting action $a$ from $s$, $Q(s, a)$ is the current measured value for taking action $a$, and $N(s, a)$ is the current visit count for $a$ from $s$. In AlphaZero, $P(s, a)$ is a probability mass function over $a$ given by our neural network.

This algorithm has worked for Deepmind, Leela, Katago, Minigo, among others. However, the algorithm has a few issues:

- Rapid Policy Convergence. With PUCT, if there are two actions with slightly different value estimates, PUCT will _always_ choose the one that is slightly stronger to visit. This leads to MCTS strongly preferring one move over another, even if their values don't reflect that disparity. This can become especially troublesome if the policy converges quickly in early training, before the net has had time to learn a good value function. See [link](https://github.com/leela-zero/leela-zero/issues/2230) and [link](https://github.com/CuriosAI/sai/issues/8) for more discussion.
- Inadequate exploration. With vanilla PUCT, the algorithm will _always_ choose the move with the highest prior to explore first. Depending on $c_{puct}$ and how $Q(s, a)$ is initialized, it may take a while for MCTS to even try a different move. AlphaZero and reproductions use Dirichlet noise at the root node of each search to try to mitigate this issue, but the center of the noise is randomly chosen, and may not always reflect a good area to search.
- Low accuracy at low visit counts. After MCTS is done, reproductions usually take either the visited counts as an empirical probability distribution, or the move with the most visits as the policy training target. If visits are too low, MCTS may not have enough time to converge on the best move, perhaps having spent many visits on a single move as a result of the PUCT formula. Even at higher visit counts, this is a problem. At inference time, Leela Zero currently selects the move with the highest LCB Q-value to mitigate this ([link](https://github.com/leela-zero/leela-zero/pull/2290)).
- Policy Improvement is not guaranteed. For a hand-wavey demonstration: if we have three actions with prior [.5, .4, .1], values [0, 0, 1], and a visit budget of 2, we are guaranteed to visit action 0 first, and are likely to visit either action 0 or 1 for our second visit. Thus, the final training target would not contain action 2, the true best move.

In 2020, Ivo Danihelka et. al. published [_Policy Improvement By Planning With Gumbel_](https://openreview.net/pdf?id=bERaNdoegnO), which introduced Gumbel MCTS and addressed the above listed issues. Very roughly, the idea is:

- Sample $k$ actions at the root.
- Find Q-values for each of the $k$ actions.
- Use $\sigma(Q_a)$ as logits for a policy probability distribution for the current root, where $\sigma$ is a monotonically increasing function.

This is _very_ rough and even a little inaccurate. The paper is relatively short and contains all the details :).

Forcing search to sample $k$ actions bakes exploration into the algorithm and removes the need for Dirichlet noise. Moreover, the sampling is guided by the current policy, and we are unlikely to insert noise into unpromising board locations. Additionally, because we use the Q-values to produce our training target, we are less likely to have the policy sharpely converge on two relatively equal actions, and we have more room to update our policy as our value function improves. Most strikingly, Gumbel MCTS guarantees a policy improvement in expectation, relative the the net's current value prediction. A proof is in the paper as well.

I use 64 visits for MCTS at the beginning of training, and gradually grow this value to 448. This is in contrast to reproductions that use visit counts in the high hundreds to low thousands.

[Paper](https://openreview.net/pdf?id=bERaNdoegnO)

## Bottleneck and Broadcast Blocks

These are new neural network blocks used in the Gumbel paper.

For background: the architecture for AlphaZero networks usually contain a _trunk_ of n blocks. Most of these blocks are convolutional, with optional intra-channel mixing blocks scattered in between.

Bottleneck blocks are an alternative type of convolutional block. A typical convolutional block contains $c$ filters, and consists of 2 x (Convolution, ReLU, and BatchNorm) blocks stacked on top of each other. A bottleneck block instead consists of a 1x1 convolution from $c$ to $c_{btl}$ channels, followed by $k$ convolutions with $c_{btl}$ channels, before a final 1x1 convolution back to $c$ channels. In my implementation, each convolution contains an associated ReLU and BatchNorm, but these may not be totally necessary.

Empirically, bottleneck blocks help with learning efficiency. The most recent Katago models use bottleneck blocks in their architecture.

Broadcast blocks are blocks that do intra-channel mixing. The core operation that a broadcast does is a fully-connected layer + ReLU operation on each channel, sharing weights across all channels. This allows for channels to see "globally", instead of relying on convolution kernels to propagate information slowly across the board. This inner kernel is sandwiched by two 1x1 convolutions. Broadcast blocks seem to be inspired by Katago GlobalPool blocks, and both achieve the same thing. In my experiments, using two broadcast blocks at tighter spacing works better than a single broadcast block. Using two broadcast blocks also significantly reduced the amount of time it took the net to stop playing first and second line moves.

## Playout Cap Randomization

Of the methods that [Katago](https://arxiv.org/pdf/1902.10565.pdf) introduced, this was the one I found the most striking.

In vanilla AlphaZero, the number of unique policy targets far outnumbers the number of unique value targets. Specifically, all moves in a single game share the same value target, so assuming 275 moves per game, the ratio of policy:value targets is 275:1. This makes value overfitting a huge issue.

Playout Cap Randomization addresses this by doing the following: instead of recording every move for training, for each move, perform a full search and record for training with probability 0.25. Otherwise, perform a fast search with significantly less playouts. This reduces the policy:value target ratio down to ~50:1 (the additional reduction in ratio is because moves are selected with lower probability in very skewed winning/losing positions).

A nice effect of this is that we can use a higher per-sample reuse factor--whereas early versions of Minigo has a reuse factor of ~.6, Katago has a reuse factor of 4. In my case, my reuse factor is slightly over 4 with few issues with value overfitting.

## Go Exploit Buffer

This is the core idea in [Targeted Search Control in AlphaZero for Effective Policy Improvement](https://arxiv.org/pdf/2302.12359.pdf).

The idea is to keep a buffer of previously-visited positions, and start games from these positions with some probability. Doing so changes the distribution of states visited, and increases the number of unique states visited throughout self-play.
