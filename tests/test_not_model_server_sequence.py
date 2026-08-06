from __future__ import annotations

import unittest

from NoTModel1Elman import prepare_sequence_request_state


class SequenceRequestPreparationTests(unittest.TestCase):
    def test_sequence_model_request_uses_explicit_history(self) -> None:
        model_entry = {
            "sequence_model": True,
            "sequence_length": 2,
            "base_feature_dim": 582,
        }
        request_data = {"state_history": [[float(index) for index in range(582)]]}

        prepared = prepare_sequence_request_state(model_entry, request_data)

        self.assertEqual(len(prepared), 2 * 582)
        self.assertEqual(prepared[:582], prepared[582:])

    def test_non_sequence_model_request_is_not_reinterpreted(self) -> None:
        model_entry = {"sequence_model": False}
        request_data = {"state_vector": [1.0, 2.0]}

        self.assertIsNone(prepare_sequence_request_state(model_entry, request_data))

    def test_sequence_model_request_requires_history(self) -> None:
        with self.assertRaises(ValueError):
            prepare_sequence_request_state(
                {"sequence_model": True, "sequence_length": 2, "base_feature_dim": 582},
                {},
            )


if __name__ == "__main__":
    unittest.main()
