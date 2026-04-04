from __future__ import annotations

import unittest

from ModelServerRouting import (
    choose_model_id_for_request,
    is_entity_model_entry,
    request_prefers_entity_payload,
    select_default_entity_model_id,
)


class ModelServerRoutingTests(unittest.TestCase):
    def test_identifies_entity_model_entries(self) -> None:
        self.assertTrue(
            is_entity_model_entry(
                {
                    "family_id": "entity_action_bc",
                    "state_schema_version": "entity_action_v1",
                }
            )
        )
        self.assertFalse(
            is_entity_model_entry(
                {
                    "family_id": "vector_joint_bc_transition_value",
                    "state_schema_version": "flat_public_v1",
                }
            )
        )

    def test_prefers_entity_payload_when_battle_state_is_present(self) -> None:
        self.assertTrue(request_prefers_entity_payload({"battle_state": {"turn": 1}}))
        self.assertFalse(request_prefers_entity_payload({"state_vector": [0.0, 1.0]}))

    def test_select_default_entity_model_id_uses_first_entity_model(self) -> None:
        model_artifacts = {
            "model1": {"family_id": "vector_joint_bc"},
            "entity_action": {"family_id": "entity_action_bc"},
            "entity_invariance": {"family_id": "entity_invariance_aux"},
        }

        self.assertEqual(
            select_default_entity_model_id(["model1", "entity_action", "entity_invariance"], model_artifacts),
            "entity_action",
        )

    def test_choose_model_id_for_request_routes_entity_payloads_to_entity_default(self) -> None:
        model_id = choose_model_id_for_request(
            {"battle_state": {"turn": 1}},
            default_model_id="model1",
            default_entity_model_id="entity_action",
            supported_model_ids=["entity_action", "model1"],
        )

        self.assertEqual(model_id, "entity_action")

    def test_choose_model_id_for_request_routes_vector_payloads_to_default_model(self) -> None:
        model_id = choose_model_id_for_request(
            {"state_vector": [0.0, 1.0]},
            default_model_id="model1",
            default_entity_model_id="entity_action",
            supported_model_ids=["entity_action", "model1"],
        )

        self.assertEqual(model_id, "model1")

    def test_choose_model_id_for_request_rejects_entity_payloads_without_entity_models(self) -> None:
        with self.assertRaises(KeyError):
            choose_model_id_for_request(
                {"battle_state": {"turn": 1}},
                default_model_id="model1",
                default_entity_model_id=None,
                supported_model_ids=["model1"],
            )


if __name__ == "__main__":
    unittest.main()
