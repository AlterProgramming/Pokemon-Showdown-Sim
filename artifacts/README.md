# Model Artifacts

Archived model artifacts for Pokemon-Showdown-Agents-Go-Brrrr.

**Primary storage: `gs://artifacts-model-serving`** — use GCS for serving and downloads.
This branch is a git-tracked mirror for version history.

## Structure

```
artifacts/
  flat_policy/          Flat vector policy models (model1–5)
                        Each model: .keras, training_model, policy_value, vocab, metadata
  entity_action_bc_v1/  Entity action BC v1 family
                        └── entity_action_bc_v1_20260408_0428/
  entity_action_v2/     Entity action v2 family
                        └── entity_action_v2_20260409_1811/
  legacy/               Pre-numbered models
  model_registry.json   Model ID → path/metadata index
```

## Model families

| Family | IDs | Architecture |
|---|---|---|
| `vector_joint_bc_v1` | model1 | Flat state, joint action vocab |
| `vector_joint_bc_transition_v1` | model2 | + turn-outcome auxiliary head |
| `vector_joint_bc_transition_value_v1` | model4, model5 | + terminal win-probability value head |
| `entity_action_bc_v1` | entity_action_bc_v1_20260408_0428 | Entity encoder, joint vocab |
| `entity_action_v2` | entity_action_v2_20260409_1811 | Entity encoder, legal-candidate conditioned |
