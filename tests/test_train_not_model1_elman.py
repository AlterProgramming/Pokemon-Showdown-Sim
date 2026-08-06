from __future__ import annotations

import unittest
from unittest.mock import patch
import sys

import numpy as np

from train_not_model1_elman import (
    alignment_loss_from_representations,
    build_not_metadata,
    build_sequence_dataset,
    masked_policy_accuracy,
    parse_args,
)


def example(
    battle_id: str,
    turn_number: int,
    player: str,
    state_vector: list[float],
    action_token: str,
) -> dict:
    return {
        "battle_id": battle_id,
        "turn_number": turn_number,
        "player": player,
        "state_vector": state_vector,
        "action_token": action_token,
    }


class SequenceDatasetTests(unittest.TestCase):
    def test_training_cli_exposes_the_reproduction_defaults(self) -> None:
        with patch.object(sys, "argv", ["train_not_model1_elman.py"]):
            args = parse_args()

        self.assertEqual(args.model_name, "model1_not_elman3")
        self.assertEqual(args.sequence_length, 16)
        self.assertEqual(args.depth, 3)
        self.assertEqual(args.hidden_dim, 256)

    def test_sequence_dataset_keeps_battle_histories_separate(self) -> None:
        examples = [
            example("battle-a", 1, "p1", [1.0, 1.0], "move:a"),
            example("battle-a", 2, "p1", [2.0, 2.0], "move:b"),
            example("battle-b", 1, "p1", [9.0, 9.0], "move:a"),
        ]

        dataset = build_sequence_dataset(
            examples,
            {"<UNK>": 0, "move:a": 1, "move:b": 2},
            sequence_length=2,
            feature_dim=2,
        )

        self.assertEqual(dataset["states"].shape, (3, 2, 2))
        np.testing.assert_array_equal(dataset["states"][0], [[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_array_equal(dataset["states"][1], [[1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_array_equal(dataset["states"][2], [[9.0, 9.0], [9.0, 9.0]])
        np.testing.assert_array_equal(dataset["masks"][0], [0.0, 1.0])
        np.testing.assert_array_equal(dataset["labels"][1], [1, 2])
        self.assertEqual(dataset["battle_ids"].tolist(), ["battle-a", "battle-a", "battle-b"])

    def test_sequence_dataset_maps_unknown_actions_to_the_vocab_unknown_class(self) -> None:
        examples = [example("battle-a", 1, "p1", [1.0, 1.0], "move:missing")]

        dataset = build_sequence_dataset(
            examples,
            {"<UNK>": 0, "move:a": 1},
            sequence_length=2,
            feature_dim=2,
        )

        self.assertEqual(int(dataset["labels"][0, -1]), 0)

    def test_masked_policy_accuracy_ignores_left_padding(self) -> None:
        logits = np.asarray(
            [
                [[10.0, 0.0], [0.0, 10.0]],
                [[9.0, 1.0], [0.0, 9.0]],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        masks = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        self.assertAlmostEqual(masked_policy_accuracy(logits, labels, masks), 0.5)

    def test_not_metadata_identifies_model1_parent_and_sequence_contract(self) -> None:
        metadata = build_not_metadata(
            guide_model_path="artifacts/model_1.keras",
            target_model_path="artifacts/model1_not_elman3.keras",
            policy_vocab_path="artifacts/action_vocab_model1_not_elman3.json",
            sequence_length=16,
            base_feature_dim=582,
            num_action_classes=357,
            train_examples=10,
            val_examples=4,
            stage_metrics=[],
        )

        self.assertEqual(metadata["model_release_id"], "model1_not_elman3")
        self.assertEqual(metadata["parent_release_id"], "model1")
        self.assertEqual(metadata["family_id"], "vector_joint_bc_not_elman")
        self.assertEqual(metadata["feature_dim"], 16 * 582)
        self.assertTrue(metadata["sequence_model"])
        self.assertEqual(metadata["sequence_padding"], "repeat_first")
        self.assertEqual(metadata["action_space"], "joint")

    def test_alignment_loss_is_zero_for_matching_valid_hidden_rows(self) -> None:
        guide = [
            np.asarray(
                [[[1.0, 0.0], [0.0, 1.0]]],
                dtype=np.float32,
            )
        ]
        target = [guide[0].copy()]
        masks = np.asarray([[1.0, 1.0]], dtype=np.float32)

        self.assertAlmostEqual(alignment_loss_from_representations(guide, target, masks), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
