from __future__ import annotations

import unittest

from word_prediction_model.replay_reconstruction import reconstruct_battle_state


class ReplayReconstructionTests(unittest.TestCase):
    def test_reconstructs_basic_state_and_rl_move_decision(self) -> None:
        log_text = "\n".join(
            [
                "|turn|1",
                "|switch|p1a: Noctowl|Noctowl, L95, M|344/344",
                "|switch|p2a: Hitmontop|Hitmontop, L88, M|231/231",
                "|move|p1a: Noctowl|Hurricane|p2a: Hitmontop|[miss]",
                "|move|p2a: Hitmontop|Close Combat|p1a: Noctowl",
                "|-damage|p1a: Noctowl|164/344",
                "|-unboost|p2a: Hitmontop|def|1",
                "|-unboost|p2a: Hitmontop|spd|1",
            ]
        )

        final_state, decision_points = reconstruct_battle_state(log_text)

        self.assertEqual(final_state["p1"]["active_uid"], "p1_noctowl")
        self.assertEqual(final_state["p2"]["active_uid"], "p2_hitmontop")
        self.assertEqual(final_state["mons"]["p1_noctowl"]["hp"], 164)
        self.assertIn("Close Combat", final_state["mons"]["p2_hitmontop"]["observed_moves"])
        self.assertEqual(len(decision_points), 1)
        self.assertEqual(decision_points[0].actual_action_id, "closecombat")
        self.assertEqual(decision_points[0].player, "p2")
        self.assertEqual(decision_points[0].battle_state["turn_index"], 1)

    def test_reconstructs_status_item_and_faint(self) -> None:
        log_text = "\n".join(
            [
                "|turn|3",
                "|switch|p1a: Weezing|Weezing-Galar, L86, M|252/252",
                "|switch|p2a: Dondozo|Dondozo, L78, M|362/362",
                "|-item|p2a: Dondozo|Leftovers",
                "|move|p1a: Weezing|Gunk Shot|p2a: Dondozo",
                "|-damage|p2a: Dondozo|19/362",
                "|-status|p2a: Dondozo|psn",
                "|move|p2a: Dondozo|Wave Crash|p1a: Weezing",
                "|-damage|p1a: Weezing|0 fnt",
                "|faint|p1a: Weezing",
                "|-damage|p2a: Dondozo|12/362 psn|[from] Recoil",
                "|-heal|p2a: Dondozo|34/362 psn|[from] item: Leftovers",
            ]
        )

        final_state, _decision_points = reconstruct_battle_state(log_text)

        self.assertEqual(final_state["mons"]["p2_dondozo"]["item"], "Leftovers")
        self.assertEqual(final_state["mons"]["p2_dondozo"]["status"], "psn")
        self.assertEqual(final_state["mons"]["p2_dondozo"]["hp"], 34)
        self.assertTrue(final_state["mons"]["p1_weezinggalar"]["fainted"])
        self.assertIsNone(final_state["p1"]["active_uid"])

    def test_reconstructs_immune_ability_reveal(self) -> None:
        log_text = "\n".join(
            [
                "|turn|16",
                "|switch|p1a: Bronzong|Bronzong, L88|261/261",
                "|switch|p2a: Gliscor|Gliscor, L76, M|239/239",
                "|move|p2a: Gliscor|Earthquake|p1a: Bronzong",
                "|-immune|p1a: Bronzong|[from] ability: Levitate",
            ]
        )

        final_state, _decision_points = reconstruct_battle_state(log_text)

        self.assertEqual(final_state["mons"]["p1_bronzong"]["ability"], "Levitate")

    def test_reconstructs_weather_state(self) -> None:
        log_text = "\n".join(
            [
                "|turn|5",
                "|switch|p1a: Ninetales|Ninetales, L85, F|263/263",
                "|switch|p2a: Tentacruel|Tentacruel, L84, M|272/272",
                "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Ninetales",
            ]
        )

        final_state, _decision_points = reconstruct_battle_state(log_text)

        self.assertEqual(final_state["field"]["weather"], "sunnyday")


if __name__ == "__main__":
    unittest.main()
