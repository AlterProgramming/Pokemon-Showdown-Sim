from __future__ import annotations

import unittest

import numpy as np

from NoTModel1Elman import (
    cka_dissimilarity,
    flatten_state_history,
    linear_cka,
    normalize_model1_state_vector,
    pad_state_history,
    progressive_replacement_schedule,
)


class NoTSequenceUtilityTests(unittest.TestCase):
    def test_progressive_schedule_adds_one_recurrent_layer_to_the_replaced_prefix(self) -> None:
        self.assertEqual(
            progressive_replacement_schedule(3),
            ((0,), (0, 1), (0, 1, 2)),
        )

    def test_progressive_schedule_rejects_negative_depth(self) -> None:
        with self.assertRaises(ValueError):
            progressive_replacement_schedule(-1)

    def test_pad_state_history_repeats_first_observation_and_masks_padding(self) -> None:
        history = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        padded, mask = pad_state_history(history, sequence_length=4, feature_dim=2)

        np.testing.assert_array_equal(
            padded,
            np.asarray(
                [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(mask, np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    def test_pad_state_history_keeps_only_the_most_recent_observations(self) -> None:
        history = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)

        padded, mask = pad_state_history(history, sequence_length=2, feature_dim=1)

        np.testing.assert_array_equal(padded, np.asarray([[2.0], [3.0]], dtype=np.float32))
        np.testing.assert_array_equal(mask, np.asarray([1.0, 1.0], dtype=np.float32))

    def test_pad_state_history_rejects_wrong_feature_dimension(self) -> None:
        with self.assertRaises(ValueError):
            pad_state_history(np.zeros((2, 3), dtype=np.float32), sequence_length=4, feature_dim=2)

    def test_linear_cka_is_one_for_identical_representations(self) -> None:
        representations = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )

        self.assertAlmostEqual(linear_cka(representations, representations), 1.0, places=6)
        self.assertAlmostEqual(cka_dissimilarity(representations, representations), 0.0, places=6)

    def test_linear_cka_penalizes_different_batch_geometry(self) -> None:
        guide = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
            dtype=np.float32,
        )
        target = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
            dtype=np.float32,
        )

        self.assertLess(linear_cka(guide, target), 0.95)
        self.assertGreater(cka_dissimilarity(guide, target), 0.05)

    def test_normalize_model1_state_vector_removes_current_slot_and_team_extensions(self) -> None:
        current_vector = list(range(690))

        normalized = normalize_model1_state_vector(current_vector)

        self.assertEqual(len(normalized), 582)
        self.assertEqual(normalized[:232], list(range(232)))
        self.assertEqual(normalized[232], 233)
        self.assertEqual(normalized[232 + 27], 261)
        self.assertEqual(normalized[556], 664)
        self.assertEqual(normalized[-1], 689)

    def test_normalize_model1_state_vector_accepts_the_historical_582_layout(self) -> None:
        vector = list(range(582))

        self.assertEqual(normalize_model1_state_vector(vector), vector)

    def test_flatten_state_history_normalizes_each_observation_and_uses_repeat_first_padding(self) -> None:
        current_vector = list(range(690))

        flattened = flatten_state_history(
            [current_vector],
            sequence_length=2,
            feature_dim=582,
        )

        self.assertEqual(len(flattened), 2 * 582)
        self.assertEqual(flattened[:582], flattened[582:])
        self.assertEqual(flattened[-1], 689)


if __name__ == "__main__":
    unittest.main()
