# Core

Shared Python modules that the training, serving, and analysis scripts all reuse.

Main areas:

- `ActionLegality.py` for legal move/switch filtering helpers.
- `BattleStateTracker.py`, `StateVectorization.py`, `StaticDex.py`, and `TrainingSplit.py` for battle-state parsing and feature generation.
- `RewardSignals.py` for reward shaping helpers.
- `ModelRegistry.py` and `ModelWorkers.py` for artifact lookup and model-serving orchestration.
