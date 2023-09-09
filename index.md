## Overview

[p3achygo](https://github.com/p3achyjr/p3achygo) is a AlphaZero-based Go engine, able to reach 5-6D after playing 500k games, for about a week of training time on an A100. The most recent run is currently ongoing. I have not currently made models/SGFs public, nor have I integrated with GTP protocols. These should all be coming soon :).

## Status

Current Run: v3 (8-15-2023 - Ongoing)

Games Played: 870,000

Current Strength: Coming Soon

## Training Efficiency

To measure p3achygo's training efficiency relative to the original AlphaGo Zero, we can compare p3achygo to [Leela Zero](https://zero.sjeng.org/). Leela Zero is an almost 100% faithful reproduction of the original AlphaGo Zero, with minor differences in visit count per move and MCTS algorithm. At 256 visits, p3achygo is slightly stronger than LZ-066 with 1000 visits. LZ-066 played ~3.2 million games, for an approximate lower bound of ~1.4T queries (Leela used [3200 visits](https://github.com/leela-zero/leela-zero/issues/1416) per move). By this measurement, p3achygo is ~140 times more efficient than LZ in early training. The original [AlphaGo Zero](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ) ended its 20-block run after ~2T queries, and 4.9 million games.

For more docs on runs and experiments, see [runs.md](https://github.com/p3achyjr/p3achygo/blob/main/notes/runs.md).
