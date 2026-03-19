from __future__ import annotations

import unittest

from ActionLegality import filter_legal_revive_targets, filter_legal_switches


class ActionLegalityTests(unittest.TestCase):
    def test_force_switch_allows_switches_even_when_trapped(self) -> None:
        legal_switches = [{"slot": 2}, {"slot": 3}]
        filtered, reason = filter_legal_switches(
            {"forceSwitch": [True], "active": [{"trapped": True}]},
            legal_switches,
        )

        self.assertEqual(filtered, legal_switches)
        self.assertEqual(reason, "force_switch")

    def test_trapped_active_blocks_voluntary_switches(self) -> None:
        filtered, reason = filter_legal_switches(
            {"active": [{"trapped": True}]},
            [{"slot": 2}, {"slot": 3}],
        )

        self.assertEqual(filtered, [])
        self.assertEqual(reason, "trapped")

    def test_side_active_trap_blocks_switches(self) -> None:
        filtered, reason = filter_legal_switches(
            {
                "side": {
                    "pokemon": [
                        {"slot": 1, "active": True, "trapped": True},
                        {"slot": 2, "active": False},
                    ]
                }
            },
            [{"slot": 2}],
        )

        self.assertEqual(filtered, [])
        self.assertEqual(reason, "trapped")

    def test_active_context_allows_switches_when_not_trapped(self) -> None:
        legal_switches = [{"slot": 2}, {"slot": 3}]
        filtered, reason = filter_legal_switches(
            {"active": [{"moves": []}]},
            legal_switches,
        )

        self.assertEqual(filtered, legal_switches)
        self.assertEqual(reason, "request_active_context")

    def test_missing_legality_context_blocks_voluntary_switches(self) -> None:
        filtered, reason = filter_legal_switches(
            {"state_vector": [0.0, 1.0]},
            [{"slot": 2}],
        )

        self.assertEqual(filtered, [])
        self.assertEqual(reason, "missing_legality_context")

    def test_assume_legal_switches_restores_old_behavior_when_requested(self) -> None:
        legal_switches = [{"slot": 2}, {"slot": 3}]
        filtered, reason = filter_legal_switches(
            {"assume_legal_switches": True},
            legal_switches,
        )

        self.assertEqual(filtered, legal_switches)
        self.assertEqual(reason, "assumed_legal")

    def test_per_switch_disabled_and_fainted_entries_are_filtered(self) -> None:
        legal_switches = [
            {"slot": 2, "disabled": True},
            {"slot": 3, "fainted": True},
            {"slot": 4, "canSwitch": True},
        ]
        filtered, reason = filter_legal_switches(
            {"switching_allowed": True},
            legal_switches,
        )

        self.assertEqual(filtered, [{"slot": 4, "canSwitch": True}])
        self.assertEqual(reason, "explicit_allow")

    def test_revive_request_prefers_fainted_targets(self) -> None:
        targets, reason = filter_legal_revive_targets(
            {"reviving": True},
            [
                {"slot": 2, "fainted": True},
                {"slot": 3, "fainted": False},
                {"slot": 4, "fainted": True},
            ],
        )

        self.assertEqual(targets, [{"slot": 2, "fainted": True}, {"slot": 4, "fainted": True}])
        self.assertEqual(reason, "revive_target_selection")

    def test_all_fainted_slot_candidates_infer_revive_request(self) -> None:
        targets, reason = filter_legal_revive_targets(
            {"state_vector": [0.0]},
            [{"slot": 2, "fainted": True}, {"slot": 3, "fainted": True}],
        )

        self.assertEqual(targets, [{"slot": 2, "fainted": True}, {"slot": 3, "fainted": True}])
        self.assertEqual(reason, "revive_target_selection")

    def test_non_revive_slot_request_does_not_emit_revive_targets(self) -> None:
        targets, reason = filter_legal_revive_targets(
            {"active": [{"moves": []}]},
            [{"slot": 2}, {"slot": 3}],
        )

        self.assertEqual(targets, [])
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
