# Release History - `apy-oracle`

## 0.8.0.post1 (2023-01-27)

- Fixes ACN configuration #35


## 0.8.0 (2023-01-26)

- Adds missing section to docs #28
- Handles past blocks errors #30
- Adds doc linters #31
- Bumps `open-autonomy` and `open-aea` and addresses breaking changes #32
- Adds release workflow #33


## 0.7.0 (2023-01-20)

- Integrates ACN #27
- Bumps `open-autonomy` and `open-aea` and addresses breaking changes #23 && #26
- Adds docs #24 && #25
- Routes IPFS traffic to dedicated cluster #22


## 0.4.0 (2022-11-22)

- Fixes reset and pause timeout for the chained FSM #20
- Utilizes the updated `ApiSpecs` #19
- Bumps `open-autonomy` and `open-aea` and addresses breaking changes #18


## 0.3.5 (2022-11-17)

- Introduces the reset and pause round to help keep the memory usage constant instead of growing unbounded #15
- Bumps `open-autonomy` and `open-aea` #13


## 0.3.4 (2022-11-08)

- Makes the APY calculation interval aware #11
- Bumps `open-autonomy` and `open-aea` #10
- Fixes duplicate estimates #9
- Reports DEX and coin names for the APY estimates #8
- Adds missing Tendermint dialogues #7


## 0.3.3 (2022-10-24)

- Bumped open-autonomy to `0.3.3` #6
- Fixed Makefile target for building release image #5
- Fixed the `NaN` predictions issue #4
- Modified tests to cover the forecasting `NaN` issue #3
