from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
import subprocess
import sys

from word_prediction_model.battle_inquiry import sample_battle_state
import word_prediction_model.policy_server as policy_server
from word_prediction_model.policy_server import app, default_question_for_state


class PolicyServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        policy_server.STATS_DIR = Path(self._tmpdir.name)
        policy_server.STATS_FLUSH_EVERY = 1
        self.client = app.test_client()
        policy_server._RESPONSE_CACHE = policy_server._ResponseCache(max_size=128)
        policy_server._SERVER_STATS = policy_server._ServerStats()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_trapped_low_hp_state_does_not_route_to_switch_question(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.08
        question = default_question_for_state(
            {
                "battle_state": state,
                "perspective_player": "p1",
                "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
                "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
                "active": [{"trapped": True}],
            }
        )

        self.assertEqual(question, "should I attack now?")

    def test_setup_question_requires_clean_setup_window(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["hp_frac"] = 0.85
        state["mons"]["p1a"]["boosts"]["atk"] = 1
        state["mons"]["p2a"]["hp_frac"] = 0.8
        question = default_question_for_state(
            {
                "battle_state": state,
                "perspective_player": "p1",
                "legal_moves": [
                    {"move": "Thunderbolt", "slot": 1},
                    {"move": "Swords Dance", "slot": 2},
                ],
                "legal_switches": [],
                "active": [{}],
            }
        )

        self.assertEqual(question, "should I attack now?")

    def test_material_advantage_routes_to_closeout_question(self) -> None:
        state = sample_battle_state()
        state["p1"]["slots"] = ["p1a", "p1b", "p1c", None, None, None]
        state["p2"]["slots"] = ["p2a", None, None, None, None, None]
        state["mons"]["p1a"]["hp_frac"] = 0.85
        state["mons"]["p1c"] = {
            "uid": "p1c",
            "species": "Squirtle",
            "hp_frac": 0.88,
            "status": None,
            "fainted": False,
            "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        }
        state["mons"]["p2a"]["hp_frac"] = 0.5
        question = default_question_for_state(
            {
                "battle_state": state,
                "perspective_player": "p1",
                "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
                "legal_switches": [],
                "active": [{}],
            }
        )

        self.assertEqual(question, "how do I close this out safely?")

    def test_low_hp_cleanup_routes_to_priority_question(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Pikachu"
        state["mons"]["p1a"]["hp_frac"] = 0.28
        state["mons"]["p2a"]["species"] = "Jolteon"
        state["mons"]["p2a"]["hp_frac"] = 0.2
        question = default_question_for_state(
            {
                "battle_state": state,
                "perspective_player": "p1",
                "legal_moves": [
                    {"move": "Quick Attack", "id": "quickattack", "slot": 1},
                    {"move": "Thunderbolt", "id": "thunderbolt", "slot": 2},
                ],
                "legal_switches": [],
                "active": [{}],
            }
        )

        self.assertEqual(question, "do I need priority here?")

    def test_revealed_threat_routes_to_preserve_switch_question(self) -> None:
        state = sample_battle_state()
        state["mons"]["p1a"]["species"] = "Charizard"
        state["mons"]["p1a"]["hp_frac"] = 0.35
        state["mons"]["p2a"]["species"] = "Blastoise"
        state["mons"]["p2a"]["hp_frac"] = 0.72
        state["mons"]["p2a"]["observed_moves"] = ["surf"]
        question = default_question_for_state(
            {
                "battle_state": state,
                "perspective_player": "p1",
                "legal_moves": [{"move": "Flamethrower", "slot": 1}],
                "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
                "active": [{}],
            }
        )

        self.assertEqual(question, "should I preserve this and switch?")

    def test_predict_batch_matches_single_predict(self) -> None:
        state_a = sample_battle_state()
        state_b = sample_battle_state()
        state_b["mons"]["p1a"]["hp_frac"] = 0.14
        state_b["mons"]["p2a"]["hp_frac"] = 0.78

        request_a = {
            "battle_state": state_a,
            "perspective_player": "p1",
            "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
            "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
            "active": [{}],
        }
        request_b = {
            "battle_state": state_b,
            "perspective_player": "p1",
            "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
            "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
            "active": [{}],
        }

        single_a = self.client.post("/predict", json=request_a)
        single_b = self.client.post("/predict", json=request_b)
        batch = self.client.post("/predict-batch", json={"requests": [request_a, request_b]})

        self.assertEqual(single_a.status_code, 200)
        self.assertEqual(single_b.status_code, 200)
        self.assertEqual(batch.status_code, 200)
        self.assertEqual(
            batch.get_json()["results"],
            [single_a.get_json(), single_b.get_json()],
        )

    def test_predict_batch_rejects_invalid_entries_with_index(self) -> None:
        response = self.client.post("/predict-batch", json={"requests": [{"perspective_player": "p1"}]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "missing battle_state")
        self.assertEqual(response.get_json()["index"], 0)

    def test_micro_batcher_coalesces_concurrent_requests(self) -> None:
        captured_batch_sizes: list[int] = []
        original_batch = policy_server.choose_actions_from_inquiry_batch
        original_single = policy_server.choose_action_from_inquiry
        try:
            def fake_batch(inquiries, *, model, top_k=3):
                captured_batch_sizes.append(len(inquiries))
                return [{"type": "move", "word_model_primary": "attack", "prompt_tokens": ["pressure"]} for _ in inquiries]

            def fake_single(*args, **kwargs):
                return {"type": "move", "word_model_primary": "attack", "prompt_tokens": ["pressure"]}

            policy_server.choose_actions_from_inquiry_batch = fake_batch
            policy_server.choose_action_from_inquiry = fake_single

            batcher = policy_server._PredictMicroBatcher(batch_window_ms=10.0, max_batch_size=8)
            base_state = sample_battle_state()
            parsed_request = {
                "question": "should I attack now?",
                "battle_state": base_state,
                "perspective_player": "p1",
                "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
                "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
                "allow_voluntary_switches": True,
            }

            results = [None] * 4

            def run(index: int) -> None:
                results[index] = batcher.submit(parsed_request)

            threads = [threading.Thread(target=run, args=(index,)) for index in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertTrue(captured_batch_sizes)
            self.assertGreaterEqual(max(captured_batch_sizes), 2)
            self.assertTrue(all(result["type"] == "move" for result in results))
        finally:
            policy_server.choose_actions_from_inquiry_batch = original_batch
            policy_server.choose_action_from_inquiry = original_single

    def test_micro_batcher_uses_cached_result_for_repeated_request(self) -> None:
        call_count = 0
        original_single = policy_server.choose_action_from_inquiry
        try:
            def fake_single(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return {"type": "move", "word_model_primary": "attack", "prompt_tokens": ["pressure"]}

            policy_server.choose_action_from_inquiry = fake_single
            batcher = policy_server._PredictMicroBatcher(batch_window_ms=0.0, max_batch_size=8)
            parsed_request = {
                "question": "should I attack now?",
                "battle_state": sample_battle_state(),
                "perspective_player": "p1",
                "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
                "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
                "allow_voluntary_switches": True,
            }

            first = batcher.submit(parsed_request)
            second = batcher.submit(parsed_request)

            self.assertEqual(first, second)
            self.assertEqual(call_count, 1)
        finally:
            policy_server.choose_action_from_inquiry = original_single

    def test_cache_key_matches_equivalent_requests(self) -> None:
        request_a = {
            "question": "should I attack now?",
            "battle_state": sample_battle_state(),
            "perspective_player": "p1",
            "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
            "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
            "allow_voluntary_switches": True,
        }
        request_b = {
            "question": "should I attack now?",
            "battle_state": sample_battle_state(),
            "perspective_player": "p1",
            "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
            "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
            "allow_voluntary_switches": True,
        }

        self.assertEqual(
            policy_server._cache_key_for_request(request_a),
            policy_server._cache_key_for_request(request_b),
        )

    def test_health_exposes_runtime_stats(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("stats", payload)
        self.assertEqual(payload["stats"]["request_count"], 0)
        self.assertIn("cache_hit_rate", payload["stats"])
        self.assertIn("worker_stats", payload)

    def test_stats_track_cache_hits_and_misses(self) -> None:
        original_single = policy_server.choose_action_from_inquiry
        try:
            def fake_single(*args, **kwargs):
                return {"type": "move", "word_model_primary": "attack", "prompt_tokens": ["pressure"]}

            policy_server.choose_action_from_inquiry = fake_single
            batcher = policy_server._PredictMicroBatcher(batch_window_ms=0.0, max_batch_size=8)
            parsed_request = {
                "question": "should I attack now?",
                "battle_state": sample_battle_state(),
                "perspective_player": "p1",
                "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
                "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
                "allow_voluntary_switches": True,
            }

            batcher.submit(parsed_request)
            batcher.submit(parsed_request)
            stats = policy_server._get_server_stats().snapshot()

            self.assertEqual(stats["request_count"], 2)
            self.assertEqual(stats["cache_misses"], 1)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["micro_batches"], 1)
            self.assertEqual(stats["micro_batch_requests"], 1)
        finally:
            policy_server.choose_action_from_inquiry = original_single

    def test_health_aggregates_stats_files(self) -> None:
        other_path = policy_server.STATS_DIR / f"word_policy_{policy_server.SERVER_PORT}_99999.json"
        other_path.write_text(
            '{"request_count": 3, "cache_hits": 1, "cache_misses": 2, "micro_batches": 2, "micro_batch_requests": 2, "uncached_compute_seconds": 0.004}',
            encoding="utf-8",
        )
        policy_server._get_server_stats().record_cache_miss()

        payload = self.client.get("/health").get_json()

        self.assertEqual(payload["stats"]["request_count"], 4)
        self.assertEqual(payload["stats"]["cache_hits"], 1)
        self.assertEqual(payload["stats"]["cache_misses"], 3)

    def test_stats_can_defer_flush_until_health(self) -> None:
        policy_server.STATS_FLUSH_EVERY = 100
        policy_server._SERVER_STATS = policy_server._ServerStats()
        stats = policy_server._get_server_stats()
        stats.record_cache_miss()
        stats_path = policy_server._stats_file_path()

        file_payload = policy_server._load_stats_payload(stats_path)
        self.assertEqual(file_payload.get("request_count", 0), 0)

        payload = self.client.get("/health").get_json()
        self.assertEqual(payload["stats"]["request_count"], 1)

    def test_handle_predict_request_matches_http_predict(self) -> None:
        request_payload = {
            "battle_state": sample_battle_state(),
            "perspective_player": "p1",
            "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
            "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
            "active": [{}],
        }

        direct_result, direct_error = policy_server.handle_predict_request(request_payload)
        http_result = self.client.post("/predict", json=request_payload)

        self.assertIsNone(direct_error)
        self.assertEqual(http_result.status_code, 200)
        self.assertEqual(direct_result, http_result.get_json())

    def test_ipc_worker_predict_protocol(self) -> None:
        worker_path = Path(__file__).resolve().parents[1] / "ipc_policy_worker.py"
        request_payload = {
            "id": "test-1",
            "type": "predict",
            "payload": {
                "battle_state": sample_battle_state(),
                "perspective_player": "p1",
                "legal_moves": [{"move": "Thunderbolt", "slot": 1}],
                "legal_switches": [{"slot": 2, "hp_frac": 0.9}],
                "active": [{}],
            },
        }

        process = subprocess.Popen(
            [sys.executable, str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(json.dumps(request_payload) + "\n")
            process.stdin.flush()
            raw_response = process.stdout.readline()
            self.assertTrue(raw_response)
            response = json.loads(raw_response)
            self.assertEqual(response["id"], "test-1")
            self.assertTrue(response["ok"])
            self.assertEqual(response["result"]["type"], "move")
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
