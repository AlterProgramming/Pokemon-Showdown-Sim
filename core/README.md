# Core

Shared Python modules that the training, serving, and analysis scripts all reuse.

Main areas:

- `ActionLegality.py` for legal move/switch filtering helpers.
- `BattleStateTracker.py`, `StateVectorization.py`, `StaticDex.py`, and `TrainingSplit.py` for battle-state parsing and feature generation.
- `RewardSignals.py` for reward shaping helpers.
- `EntityActionV1.py`, `EntityTensorization.py`, and `EntityModelV1.py` for the first entity-centric action-scoring family.
- `EntityInvarianceTensorization.py` and `EntityInvarianceModelV1.py` for the identity-invariance follow-on family.
- `ModelRegistry.py` and `ModelWorkers.py` for artifact lookup and model-serving orchestration.
- `TransferLearning.py` for warm-start and initialization metadata helpers shared by trainers.
