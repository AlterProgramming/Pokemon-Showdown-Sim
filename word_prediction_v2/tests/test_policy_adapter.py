from __future__ import annotations

import unittest

from word_prediction_model.battle_inquiry import sample_battle_state
from word_prediction_model.policy_adapter import (
    _active_runtime_move_info,
    _state_move_score,
    choose_action_from_inquiry,
)


class PolicyAdapterTests(unittest.TestCase):
    def test_switch_word_can_trigger_switch_choice(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.05
        result = choose_action_from_inquiry(
            "should I switch out here?",
            state,
            legal_moves=[{"move": "thunderbolt", "slot": 1}],
            legal_switches=[{"slot": 2, "hp_frac": 0.94}],
        )

        self.assertEqual(result["type"], "switch")
        self.assertEqual(result["slot"], 2)

    def test_switch_word_does_not_switch_too_early(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.25
        result = choose_action_from_inquiry(
            "should I switch out here?",
            state,
            legal_moves=[{"move": "thunderbolt", "slot": 1}],
            legal_switches=[{"slot": 2, "hp_frac": 0.94}],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_trapped_active_cannot_voluntarily_switch(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.05
        result = choose_action_from_inquiry(
            "should I switch out here?",
            state,
            legal_moves=[{"move": "Thunderbolt", "slot": 1}],
            legal_switches=[{"slot": 2, "hp_frac": 0.94}],
            allow_voluntary_switches=False,
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_finish_word_prefers_attacking_move(self) -> None:
        state = sample_battle_state()
        state["mons"]["p2a"]["hp_frac"] = 0.1
        result = choose_action_from_inquiry(
            "can I knock it out now?",
            state,
            legal_moves=[
                {"move": "Thunderbolt", "slot": 1},
                {"move": "Swords Dance", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_attack_word_prefers_stab_super_effective_move(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p2a"]["species"] = "Squirtle"
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Quick Attack", "id": "quickattack", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_attack_word_prefers_stronger_damaging_move_over_hazard(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Golem"
        state["mons"]["p2a"]["species"] = "Charizard"
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Stealth Rock", "id": "stealthrock", "slot": 1},
                {"move": "Rock Slide", "id": "rockslide", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_known_super_effective_threat_prefers_recovery(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Charizard"
        state["mons"]["p1a"]["hp_frac"] = 0.28
        state["mons"]["p2a"]["species"] = "Blastoise"
        state["mons"]["p2a"]["observed_moves"] = ["surf"]
        result = choose_action_from_inquiry(
            "what is the safe play?",
            state,
            legal_moves=[
                {"move": "Flamethrower", "id": "flamethrower", "slot": 1},
                {"move": "Roost", "id": "roost", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_preserve_mode_can_choose_defensive_switch(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Charizard"
        state["mons"]["p1a"]["hp_frac"] = 0.34
        state["mons"]["p1b"]["species"] = "Venusaur"
        state["mons"]["p1b"]["hp_frac"] = 0.92
        state["mons"]["p2a"]["species"] = "Blastoise"
        state["mons"]["p2a"]["hp_frac"] = 0.72
        state["mons"]["p2a"]["observed_moves"] = ["surf"]
        result = choose_action_from_inquiry(
            "should I preserve this and switch?",
            state,
            legal_moves=[
                {"move": "Flamethrower", "id": "flamethrower", "slot": 1},
                {"move": "Roost", "id": "roost", "slot": 2},
            ],
            legal_switches=[{"slot": 2, "hp_frac": 0.92}],
        )

        self.assertEqual(result["type"], "switch")
        self.assertEqual(result["slot"], 2)

    def test_force_switch_prefers_bench_with_immediate_progress(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", "p1c", None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1b"] = {"species": "Oricorio-Pom-Pom", "hp_frac": 1.0, "status": ""}
        state["mons"]["p1c"] = {"species": "Tauros-Paldea-Combat", "hp_frac": 1.0, "status": ""}
        state["mons"]["p2a"]["species"] = "Kyogre"
        state["mons"]["p2a"]["hp_frac"] = 0.91
        result = choose_action_from_inquiry(
            "should I switch out here?",
            state,
            legal_moves=[],
            legal_switches=[
                {
                    "slot": 2,
                    "hp_frac": 1.0,
                    "species": "Oricorio-Pom-Pom",
                    "observed_moves": ["Roost", "Revelation Dance", "Hurricane", "Quiver Dance"],
                },
                {
                    "slot": 3,
                    "hp_frac": 1.0,
                    "species": "Tauros-Paldea-Combat",
                    "observed_moves": ["Close Combat", "Stone Edge", "Bulk Up", "Iron Head"],
                },
            ],
        )

        self.assertEqual(result["type"], "switch")
        self.assertEqual(result["slot"], 2)

    def test_emergency_switch_escapes_farming_matchup(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", None, None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1a"]["species"] = "Carbink"
        state["mons"]["p1a"]["hp_frac"] = 0.88
        state["mons"]["p1a"]["observed_moves"] = ["Moonblast", "Rest"]
        state["mons"]["p1b"]["species"] = "Charizard"
        state["mons"]["p1b"]["hp_frac"] = 0.92
        state["mons"]["p2a"]["species"] = "Bellossom"
        state["mons"]["p2a"]["hp_frac"] = 0.86
        state["mons"]["p2a"]["observed_moves"] = ["Strength Sap", "Giga Drain", "Quiver Dance", "Moonblast"]
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Moonblast", "id": "moonblast", "slot": 1},
                {"move": "Rest", "id": "rest", "slot": 2},
            ],
            legal_switches=[{"slot": 2, "hp_frac": 0.92}],
        )

        self.assertEqual(result["type"], "switch")
        self.assertEqual(result["slot"], 2)

    def test_emergency_switch_does_not_override_good_attack(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", None, None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.72
        state["mons"]["p1b"]["species"] = "Bulbasaur"
        state["mons"]["p1b"]["hp_frac"] = 0.88
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.82
        state["mons"]["p2a"]["observed_moves"] = ["Recover"]
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 1},
                {"move": "Quick Attack", "id": "quickattack", "slot": 2},
            ],
            legal_switches=[{"slot": 2, "hp_frac": 0.88}],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_emergency_switch_does_not_override_hazard_progress(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", None, None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1a"]["species"] = "Vikavolt"
        state["mons"]["p1a"]["hp_frac"] = 0.21
        state["mons"]["p1b"]["species"] = "Lilligant"
        state["mons"]["p1b"]["hp_frac"] = 1.0
        state["mons"]["p2a"]["species"] = "Jirachi"
        state["mons"]["p2a"]["hp_frac"] = 0.67
        state["mons"]["p2a"]["observed_moves"] = ["Wish", "Protect", "Psychic"]
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Sticky Web", "id": "stickyweb", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[{"slot": 2, "hp_frac": 1.0}],
        )

        self.assertEqual(result["type"], "move")

    def test_emergency_switch_does_not_override_knock_off_progress(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", None, None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1a"]["species"] = "Spiritomb"
        state["mons"]["p1a"]["hp_frac"] = 0.41
        state["mons"]["p1b"]["species"] = "Torkoal"
        state["mons"]["p1b"]["hp_frac"] = 1.0
        state["mons"]["p2a"]["species"] = "Sylveon"
        state["mons"]["p2a"]["hp_frac"] = 0.78
        state["mons"]["p2a"]["item"] = "Leftovers"
        state["mons"]["p2a"]["observed_moves"] = ["Wish", "Protect", "Calm Mind", "Hyper Voice"]
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Shadow Ball", "id": "shadowball", "slot": 1},
                {"move": "Knock Off", "id": "knockoff", "slot": 2},
            ],
            legal_switches=[{"slot": 2, "hp_frac": 1.0}],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_preserve_mode_respects_sack_value_closeout(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", None, None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1a"]["species"] = "Mimikyu"
        state["mons"]["p1a"]["hp_frac"] = 0.32
        state["mons"]["p1a"]["observed_moves"] = ["Shadow Claw"]
        state["mons"]["p1b"]["species"] = "Palafin"
        state["mons"]["p1b"]["hp_frac"] = 1.0
        state["mons"]["p2a"]["species"] = "Yanmega"
        state["mons"]["p2a"]["hp_frac"] = 0.44
        state["mons"]["p2a"]["observed_moves"] = ["Bug Buzz"]
        result = choose_action_from_inquiry(
            "should I preserve this and switch?",
            state,
            legal_moves=[{"move": "Shadow Claw", "id": "shadowclaw", "slot": 1}],
            legal_switches=[{"slot": 2, "hp_frac": 1.0}],
        )

        self.assertEqual(result["type"], "move")

    def test_knock_off_is_preferred_into_revealed_sustain_item(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Spiritomb"
        state["mons"]["p2a"]["species"] = "Sylveon"
        state["mons"]["p2a"]["hp_frac"] = 0.78
        state["mons"]["p2a"]["item"] = "Leftovers"
        state["mons"]["p2a"]["observed_moves"] = ["Wish", "Protect", "Calm Mind", "Hyper Voice"]
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Shadow Ball", "id": "shadowball", "slot": 1},
                {"move": "Knock Off", "id": "knockoff", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_iron_defense_body_press_mirror_prefers_setup_before_iron_head(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Registeel"
        state["mons"]["p1a"]["hp_frac"] = 0.6
        state["mons"]["p1a"]["observed_moves"] = ["Body Press", "Iron Defense", "Iron Head"]
        state["mons"]["p2a"]["species"] = "Regirock"
        state["mons"]["p2a"]["hp_frac"] = 0.84
        state["mons"]["p2a"]["observed_moves"] = ["Iron Defense", "Body Press"]
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Body Press", "id": "bodypress", "slot": 1},
                {"move": "Iron Head", "id": "ironhead", "slot": 2},
                {"move": "Iron Defense", "id": "irondefense", "slot": 3},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 3)

    def test_closeout_mode_prefers_accurate_attack_over_setup(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", "p1c", None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.82
        state["mons"]["p1b"]["species"] = "Bulbasaur"
        state["mons"]["p1b"]["hp_frac"] = 0.88
        state["mons"]["p1c"] = {
            "uid": "p1c",
            "species": "Squirtle",
            "hp_frac": 0.93,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.28
        result = choose_action_from_inquiry(
            "how do I close this out safely?",
            state,
            legal_moves=[
                {"move": "Thunder", "id": "thunder", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
                {"move": "Swords Dance", "id": "swordsdance", "slot": 3},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_hazard_clear_is_preferred_when_our_side_is_loaded(self) -> None:
        state = sample_battle_state()
        state["p1"]["side_conditions"] = {"stealthrock": 1, "spikes": 2}
        state["p1"]["slots"] = ["p1a", "p1b", "p1c", None, None, None]
        state["mons"]["p1c"] = {
            "uid": "p1c",
            "species": "Squirtle",
            "hp_frac": 0.78,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        result = choose_action_from_inquiry(
            "should I clear hazards now?",
            state,
            legal_moves=[
                {"move": "Rapid Spin", "id": "rapidspin", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_hazard_setup_is_preferred_in_long_game_window(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", "p1c", "p1d", None, None]
        state["p2"]["slots"] = ["p2a", "p2b", "p2c", "p2d", None, None]
        state["mons"]["p1a"]["species"] = "Golem"
        state["mons"]["p1a"]["hp_frac"] = 0.86
        state["mons"]["p1c"] = {
            "uid": "p1c",
            "species": "Bulbasaur",
            "hp_frac": 0.82,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        state["mons"]["p1d"] = {
            "uid": "p1d",
            "species": "Squirtle",
            "hp_frac": 0.76,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        state["mons"]["p2b"] = {
            "uid": "p2b",
            "species": "Pikachu",
            "hp_frac": 0.8,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        state["mons"]["p2c"] = {
            "uid": "p2c",
            "species": "Charmander",
            "hp_frac": 0.82,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        state["mons"]["p2d"] = {
            "uid": "p2d",
            "species": "Bulbasaur",
            "hp_frac": 0.79,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        result = choose_action_from_inquiry(
            "should I get hazards up?",
            state,
            legal_moves=[
                {"move": "Stealth Rock", "id": "stealthrock", "slot": 1},
                {"move": "Rock Tomb", "id": "rocktomb", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_deny_setup_prefers_phaze_or_clear_stat_move(self) -> None:
        state = sample_battle_state()
        state["mons"]["p2a"]["boosts"]["atk"] = 2
        state["mons"]["p2a"]["observed_moves"] = ["swordsdance"]
        result = choose_action_from_inquiry(
            "should I deny setup now?",
            state,
            legal_moves=[
                {"move": "Roar", "id": "roar", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_stop_recovery_prefers_status_denial(self) -> None:
        state = sample_battle_state()
        state["mons"]["p2a"]["hp_frac"] = 0.78
        state["mons"]["p2a"]["observed_moves"] = ["recover"]
        result = choose_action_from_inquiry(
            "how do I stop recovery here?",
            state,
            legal_moves=[
                {"move": "Toxic", "id": "toxic", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_finish_window_prefers_safe_priority_cleanup(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.22
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.18
        result = choose_action_from_inquiry(
            "can I knock it out now?",
            state,
            legal_moves=[
                {"move": "Thunder", "id": "thunder", "slot": 1},
                {"move": "Quick Attack", "id": "quickattack", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_finish_window_prefers_attack_over_recover(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.16
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.2
        result = choose_action_from_inquiry(
            "how do I close this out safely?",
            state,
            legal_moves=[
                {"move": "Roost", "id": "roost", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_finish_window_prefers_accurate_cleanup_over_inaccurate_nuke(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.31
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.22
        result = choose_action_from_inquiry(
            "can I knock it out now?",
            state,
            legal_moves=[
                {"move": "Thunder", "id": "thunder", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)


    def test_closeout_window_prefers_stronger_attack_over_weak_priority(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.52
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.31
        result = choose_action_from_inquiry(
            "how do I close this out safely?",
            state,
            legal_moves=[
                {"move": "Quick Attack", "id": "quickattack", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_priority_word_prefers_priority_cleanup(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.22
        state["mons"]["p2a"]["species"] = "Jolteon"
        state["mons"]["p2a"]["hp_frac"] = 0.18
        result = choose_action_from_inquiry(
            "do I need priority here?",
            state,
            legal_moves=[
                {"move": "Quick Attack", "id": "quickattack", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_safe_ko_word_prefers_accurate_attack(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.28
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.24
        result = choose_action_from_inquiry(
            "can I get a safe knockout here?",
            state,
            legal_moves=[
                {"move": "Thunder", "id": "thunder", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_revealed_status_pressure_prefers_attack_over_recover(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.39
        state["mons"]["p2a"]["species"] = "Bulbasaur"
        state["mons"]["p2a"]["hp_frac"] = 0.63
        state["mons"]["p2a"]["observed_moves"] = ["toxic", "leechseed"]
        result = choose_action_from_inquiry(
            "what is the safe play?",
            state,
            legal_moves=[
                {"move": "Recover", "id": "recover", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_known_immunity_and_repeat_penalty_breaks_move_spam(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Golem"
        state["mons"]["p1a"]["observed_moves"] = ["Earthquake"]
        state["mons"]["p2a"]["species"] = "Bronzong"
        state["mons"]["p2a"]["ability"] = "Levitate"
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Earthquake", "id": "earthquake", "slot": 1},
                {"move": "Fire Punch", "id": "firepunch", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_wind_rider_blocks_hurricane(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Oricorio-Sensu"
        state["mons"]["p1a"]["observed_moves"] = ["Hurricane"]
        state["mons"]["p2a"]["species"] = "Brambleghast"
        state["mons"]["p2a"]["ability"] = "Wind Rider"
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Hurricane", "id": "hurricane", "slot": 1},
                {"move": "Shadow Ball", "id": "shadowball", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_self_drop_move_is_penalized_when_pp_depleted_and_spa_lowered(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Latias"
        state["mons"]["p1a"]["boosts"]["spa"] = -2
        state["mons"]["p2a"]["species"] = "Blastoise"
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Draco Meteor", "id": "dracometeor", "slot": 1},
                {"move": "Psychic", "id": "psychic", "slot": 2},
            ],
            legal_switches=[],
            active_payload=[
                {
                    "moves": [
                        {"move": "Draco Meteor", "id": "dracometeor", "pp": 2, "maxpp": 8},
                        {"move": "Psychic", "id": "psychic", "pp": 16, "maxpp": 16},
                    ]
                }
            ],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_repeated_knock_off_without_item_prefers_alternative_attack(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Hariyama"
        state["mons"]["p1a"]["observed_moves"] = ["Knock Off"]
        state["mons"]["p2a"]["species"] = "Snorlax"
        state["mons"]["p2a"]["item"] = ""
        result = choose_action_from_inquiry(
            "should I attack now?",
            state,
            legal_moves=[
                {"move": "Knock Off", "id": "knockoff", "slot": 1},
                {"move": "Drain Punch", "id": "drainpunch", "slot": 2},
            ],
            legal_switches=[],
            active_payload=[
                {
                    "moves": [
                        {"move": "Knock Off", "id": "knockoff", "pp": 14, "maxpp": 24},
                        {"move": "Drain Punch", "id": "drainpunch", "pp": 16, "maxpp": 16},
                    ]
                }
            ],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_repeated_knock_off_without_item_scores_worse_than_item_knock_off(self) -> None:
        active_info = _active_runtime_move_info(
            [
                {
                    "moves": [
                        {"move": "Knock Off", "id": "knockoff", "pp": 14, "maxpp": 24},
                        {"move": "Drain Punch", "id": "drainpunch", "pp": 16, "maxpp": 16},
                    ]
                }
            ]
        )
        move = {"move": "Knock Off", "id": "knockoff", "slot": 1}
        base_kwargs = {
            "move_payload": move,
            "my_hp": 0.82,
            "my_status": "",
            "opp_hp": 0.76,
            "opp_status": "",
            "total_boost": 0,
            "my_species": "Hariyama",
            "opp_species": "Snorlax",
            "opp_observed_moves": [],
            "my_observed_moves": ["Knock Off"],
            "boosts": {},
            "active_move_info": active_info,
            "material_edge": 0.0,
            "best_attack_pressure": 3.8,
            "my_hazard_pressure": 0.0,
            "opp_hazard_pressure": 0.0,
            "my_remaining": 4,
            "opp_remaining": 4,
            "opp_ability": "",
        }

        no_item_score = _state_move_score(opp_item="", **base_kwargs)
        item_score = _state_move_score(opp_item="Leftovers", **base_kwargs)

        self.assertLess(no_item_score, item_score)

    def test_closeout_keeps_safe_non_priority_over_priority(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Cacturne"
        state["mons"]["p2a"]["species"] = "Kyogre"
        state["mons"]["p2a"]["hp_frac"] = 0.13
        result = choose_action_from_inquiry(
            "can I knock it out now?",
            state,
            legal_moves=[
                {"move": "Seed Bomb", "id": "seedbomb", "slot": 1},
                {"move": "Sucker Punch", "id": "suckerpunch", "slot": 2},
                {"move": "Knock Off", "id": "knockoff", "slot": 3},
            ],
            legal_switches=[],
            active_payload=[
                {
                    "moves": [
                        {"move": "Seed Bomb", "id": "seedbomb", "pp": 24, "maxpp": 24},
                        {"move": "Sucker Punch", "id": "suckerpunch", "pp": 8, "maxpp": 8},
                        {"move": "Knock Off", "id": "knockoff", "pp": 24, "maxpp": 24},
                    ]
                }
            ],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)

    def test_closeout_prefers_priority_over_risky_close_combat(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Conkeldurr"
        state["mons"]["p2a"]["species"] = "Weavile"
        state["mons"]["p2a"]["hp_frac"] = 0.20
        result = choose_action_from_inquiry(
            "can I knock it out now?",
            state,
            legal_moves=[
                {"move": "Close Combat", "id": "closecombat", "slot": 1},
                {"move": "Mach Punch", "id": "machpunch", "slot": 2},
            ],
            legal_switches=[],
            active_payload=[
                {
                    "moves": [
                        {"move": "Close Combat", "id": "closecombat", "pp": 8, "maxpp": 8},
                        {"move": "Mach Punch", "id": "machpunch", "pp": 16, "maxpp": 16},
                    ]
                }
            ],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_revealed_recovery_pressure_prefers_attack_over_setup(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.74
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.68
        state["mons"]["p2a"]["observed_moves"] = ["recover"]
        result = choose_action_from_inquiry(
            "should I setup here?",
            state,
            legal_moves=[
                {"move": "Swords Dance", "id": "swordsdance", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_statused_active_prefers_attack_over_setup(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.62
        state["mons"]["p1a"]["status"] = "tox"
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.58
        result = choose_action_from_inquiry(
            "should I setup here?",
            state,
            legal_moves=[
                {"move": "Swords Dance", "id": "swordsdance", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_statused_active_prefers_attack_over_recover_when_pressure_exists(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.36
        state["mons"]["p1a"]["status"] = "tox"
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.52
        state["mons"]["p2a"]["observed_moves"] = ["toxic"]
        result = choose_action_from_inquiry(
            "what is the safe play?",
            state,
            legal_moves=[
                {"move": "Recover", "id": "recover", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_already_boosted_active_prefers_attack_over_repeat_setup(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.78
        state["mons"]["p1a"]["boosts"]["atk"] = 2
        state["mons"]["p2a"]["species"] = "Squirtle"
        state["mons"]["p2a"]["hp_frac"] = 0.72
        result = choose_action_from_inquiry(
            "should I setup here?",
            state,
            legal_moves=[
                {"move": "Swords Dance", "id": "swordsdance", "slot": 1},
                {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 2)

    def test_setup_word_prefers_setup_move(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.9
        result = choose_action_from_inquiry(
            "should I setup here?",
            state,
            legal_moves=[
                {"move": "Thunderbolt", "slot": 1},
                {"move": "Swords Dance", "slot": 2},
            ],
            legal_switches=[],
        )

        self.assertEqual(result["type"], "move")
        self.assertEqual(result["best_move"]["slot"], 1)
