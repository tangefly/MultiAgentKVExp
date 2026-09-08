import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
from agent.llm import LLMClient, NoSubAgentKVError
from scripts.browsecomp.run_browsecomp import run_with_kv_retries


class KVRetryTest(unittest.TestCase):
    def test_retry_then_success(self):
        with patch("scripts.browsecomp.run_browsecomp.run_one",
                   side_effect=[NoSubAgentKVError("missing"), {"valid_pair": True}]) as run_one:
            result = run_with_kv_retries({"query_id": "q"}, 0, SimpleNamespace())
        self.assertEqual(run_one.call_count, 2)
        self.assertEqual(result["attempts"], 2)
        self.assertTrue(result["valid_pair"])

    def test_skip_after_three_attempts_even_when_continue_disabled(self):
        with patch("scripts.browsecomp.run_browsecomp.run_one",
                   side_effect=NoSubAgentKVError("missing")) as run_one:
            result = run_with_kv_retries({"query_id": "q"}, 0,
                                         SimpleNamespace(continue_on_error=False))
        self.assertEqual(run_one.call_count, 3)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["attempts"], 3)
        self.assertNotIn("branches", result)

    def test_other_errors_are_not_retried(self):
        with patch("scripts.browsecomp.run_browsecomp.run_one", side_effect=RuntimeError("other")) as run_one:
            with self.assertRaises(RuntimeError):
                run_with_kv_retries({}, 0, SimpleNamespace())
        self.assertEqual(run_one.call_count, 1)

    def test_only_missing_kv_http_error_is_classified_for_retry(self):
        client = LLMClient()
        client.session_id = "existing"
        for detail, expected in [("No SubAgent KV matched the final prompt; cannot run a valid reuse experiment", NoSubAgentKVError),
                                 ("paired_final refuses prompt truncation", requests.HTTPError)]:
            response = requests.Response()
            response.status_code = 400
            import json
            response._content = json.dumps({"detail": detail}).encode()
            with patch.object(client, "_post", side_effect=requests.HTTPError(response=response)):
                with self.assertRaises(expected):
                    client.paired_final([], trace=["main", "sub", "main"])

    def test_zero_actual_graft_is_retryable(self):
        client = LLMClient()
        client.session_id = "existing"
        with patch.object(client, "_post", return_value={"object": "paired.chat.completion",
                "branches": {"kv_reuse": {"grafted_tokens": 0}}}):
            with self.assertRaises(NoSubAgentKVError):
                client.paired_final([], trace=["main", "sub", "main"])
