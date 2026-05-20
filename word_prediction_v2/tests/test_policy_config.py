from __future__ import annotations

import json
import os
import tempfile
import unittest

from word_prediction_model.policy_config import (
    PolicyConfig,
    clear_policy_config_cache,
    get_active_policy_config,
    policy_config_from_profile_name,
    policy_profile_names,
)


class PolicyConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("WORD_POLICY_PROFILE_NAME", None)
        os.environ.pop("WORD_POLICY_PROFILE_PATH", None)
        clear_policy_config_cache()

    def test_baseline_profile_matches_default_config(self) -> None:
        self.assertEqual(policy_config_from_profile_name("baseline"), PolicyConfig())

    def test_named_profiles_are_available(self) -> None:
        names = policy_profile_names()
        self.assertIn("baseline", names)
        self.assertIn("aggressive_closeout", names)

    def test_active_profile_comes_from_env(self) -> None:
        os.environ["WORD_POLICY_PROFILE_NAME"] = "aggressive_closeout"
        clear_policy_config_cache()

        config = get_active_policy_config()

        self.assertEqual(config.attack_low_opp_hp_bonus, 3.0)
        self.assertEqual(config.attack_priority_cleanup_bonus, 1.4)

    def test_profile_path_overrides_named_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "profile.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "attack_low_opp_hp_bonus": 3.7,
                        "recover_default_penalty": -1.25,
                    },
                    handle,
                )
            os.environ["WORD_POLICY_PROFILE_NAME"] = "baseline"
            os.environ["WORD_POLICY_PROFILE_PATH"] = path
            clear_policy_config_cache()

            config = get_active_policy_config()

            self.assertEqual(config.attack_low_opp_hp_bonus, 3.7)
            self.assertEqual(config.recover_default_penalty, -1.25)


if __name__ == "__main__":
    unittest.main()
