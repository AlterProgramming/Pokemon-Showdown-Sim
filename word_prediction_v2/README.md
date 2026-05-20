# Word Prediction Model

This project prototypes a word-level prediction model where:

- the target is a single word
- the word can appear with misspellings
- the model learns an embedding for each canonical target word
- a short prompt of descriptor words is mapped into that same embedding space

## Design

The prototype uses two learned pathways:

1. A prompt encoder that averages learned embeddings for prompt tokens.
2. A surface-form encoder that averages learned character n-gram embeddings for a target word or misspelling.

Each canonical target word has its own trainable embedding. Training optimizes:

- prompt classification over the canonical vocabulary
- alignment between misspelled surface forms and the canonical word embedding

This makes misspellings part of the training signal instead of a preprocessing error case.

## Layout

- `text.py` - normalization, typo generation, character n-grams
- `lexicon.py` - seed concept inventory and misspelling expansion
- `dataset.py` - synthetic example generation and vocab building
- `model.py` - NumPy training loop and inference
- `pipeline.py` - artifact writing, evaluation, and model reload
- `train.py` - CLI training entrypoint
- `benchmark.py` - compact benchmark-style report for a saved run
- `battle_lexicon.py` - battle-oriented response vocabulary
- `battle_inquiry.py` - map battle inquiries and battle state into word-model outputs
- `bridge_inquiry.py` - bridge-facing wrapper that returns compact structured inquiry output
- `policy_config.py` - named decoder profiles and env/file-based policy tuning
- `sweep_profiles.py` - IPC benchmark harness for comparing policy profiles
- `demo.py` - runnable demonstration

## Docs

- [FINDINGS.md](/Users/AI-CCORE/alter-programming/word_prediction_model/FINDINGS.md) - project findings across evolution, data/dials, and implementation spine
- [BENCHMARK_CHRONOLOGY.md](/Users/AI-CCORE/alter-programming/word_prediction_model/BENCHMARK_CHRONOLOGY.md) - main benchmark phases and what changed
- [PARAMETERS.md](/Users/AI-CCORE/alter-programming/word_prediction_model/PARAMETERS.md) - parameter and dial reference by layer
- [DECISION_FLOW.md](/Users/AI-CCORE/alter-programming/word_prediction_model/DECISION_FLOW.md) - simulator-to-action decision path
- [RUNBOOK_7802_OF_10000.md](/Users/AI-CCORE/alter-programming/word_prediction_model/RUNBOOK_7802_OF_10000.md) - reproduction guide for the `78.02%` benchmarked profile
- [THREAT_AWARE_NOTES.md](/Users/AI-CCORE/alter-programming/word_prediction_model/THREAT_AWARE_NOTES.md) - notes on the revealed-move threat layer and the `85.50%` qualification profile
- [ROADMAP_90_PERCENT.md](/Users/AI-CCORE/alter-programming/word_prediction_model/ROADMAP_90_PERCENT.md) - plan for the next ceiling beyond the threat-aware decoder
- [RESEARCH_SNAPSHOT_2026_04_09.md](/Users/AI-CCORE/alter-programming/word_prediction_model/RESEARCH_SNAPSHOT_2026_04_09.md) - current stable benchmark, recent decoder findings, and the latest ceiling read

## Run

```bash
python3 -m word_prediction_model.demo
```

## Train

```bash
python3 -m word_prediction_model.train --run-name local_smoke
```

This writes a testable run directory under `artifacts/word_prediction_model/` with:

- `model.json`
- `training_metadata.json`
- `evaluation_summary.json`
- `run_manifest.json`

## Benchmark

```bash
python3 -m word_prediction_model.benchmark artifacts/word_prediction_model/local_smoke
```

This prints compact benchmark metrics for a saved run, including training ticks,
overall accuracy, prompt accuracy, and surface-form accuracy.

## Battle Inquiry

```bash
python3 -m word_prediction_model.battle_inquiry --question "should I switch out here?"
```

This trains a battle-oriented word model locally, derives prompt tokens from the
question plus battle-state signals, and returns the top battle words such as
`switch`, `finish`, `stabilize`, or `attack`.

## Bridge Wrapper

```bash
python3 -m word_prediction_model.bridge_inquiry --question "should I switch out here?" --payload-file payload.json
```

This accepts a normalized payload file and returns structured JSON with:

- `primary_word`
- `primary_score`
- `backup_words`
- `prompt_tokens`
- `model_source`

## Test

```bash
python3 -m pytest word_prediction_model/tests
```

## Profile Sweeps

Use the IPC transport for tuning so the benchmark hits the Python policy codepath
rather than the older in-process JS local policy.

```bash
python3 -m word_prediction_model.sweep_profiles \
  --profiles baseline aggressive_closeout anti_loop_light safe_conversion \
  --games 100 \
  --concurrency 5
```

This writes a timestamped sweep directory under
`artifacts/word_prediction_model/sweeps/` containing:

- one JSON profile file per tested policy
- one benchmark log per tested policy
- `results.json` with ranked outcomes and parsed metrics

## Live Benchmark

Current best validated large-sample profile:

- `sticky_move_run_003`
- `1768 / 2000`
- `88.40%`
- `0` failed games

Recorded benchmark:

- [sticky_move_run_003/benchmark.log](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/recorded_runs/sticky_move_run_003/benchmark.log)

The latest research snapshot for this branch and the failed follow-on
closeout experiments is:

- [RESEARCH_SNAPSHOT_2026_04_09.md](/Users/AI-CCORE/alter-programming/word_prediction_model/RESEARCH_SNAPSHOT_2026_04_09.md)

Historical large-scale benchmark log:

- [word_policy_10000.log](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/benchmarks/word_policy_10000.log)
