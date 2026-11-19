# gta6-trophy-rarity-forecast-01
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zenganteng0987/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zenganteng0987/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI trophy's earn rate on Exophase at the target instant, as five quantiles on the logit scale.

Model: the read history is collapsed into distinct page states (the page freezes between Exophase refreshes). With a rich history, players tracked are extrapolated to the target and each trophy gets two estimates — an earners-elasticity fit against players tracked and a logit-rate trend fit — with slopes shrunk toward the rarity-tier mean; the median averages the two on the logit scale. With a frozen history, a refresh is overdue, so rates shift down by the gap between earners growth and players growth over the page age plus the horizon, with a wide spread. A trophy already at 100% stays put. Python 3.13 standard library only, no network, deterministic.

Run locally:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
