from __future__ import annotations

from .dataset import build_training_corpus
from .model import TrainingConfig, train_model


def main() -> None:
    corpus = build_training_corpus(repeats_per_entry=72, seed=23)
    model = train_model(
        corpus,
        TrainingConfig(
            embedding_dim=28,
            epochs=120,
            learning_rate=0.06,
            alignment_weight=0.45,
            seed=11,
        ),
    )

    prompts = [
        ["joy", "bright", "smile"],
        ["water", "deep", "blue"],
        ["error", "digital", "noise"],
        ["storm", "electric", "flash"],
    ]
    spellings = ["happee", "oceon", "gltich", "thundr"]

    print("Prompt predictions")
    for prompt in prompts:
        ranked = model.predict(prompt, top_k=3)
        print(f"  {prompt} -> {[f'{item.word}:{item.score:.3f}' for item in ranked]}")

    print("\nSurface-form nearest words")
    for spelling in spellings:
        ranked = model.nearest_words_for_surface_form(spelling, top_k=3)
        print(f"  {spelling} -> {[f'{item.word}:{item.score:.3f}' for item in ranked]}")


if __name__ == "__main__":
    main()
