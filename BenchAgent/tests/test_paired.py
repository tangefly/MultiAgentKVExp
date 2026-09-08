import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from agent.agent import Agent
from agent.tools import build_subagent_tools


class PairedAgentTest(unittest.TestCase):
    def make_agent(self, count=2):
        calls = [{"id": str(i), "type": "function", "function": {"name": "call_subagent",
                  "arguments": json.dumps({"task": f"document_index: {i}\ndocument_path: /doc{i}"})}}
                 for i in range(count)]
        client = Mock()
        client.chat.return_value = {"role": "assistant", "content": "<think>plan</think>", "tool_calls": calls}
        client.paired_final.return_value = {"branches": {}}
        agent = Agent("main", "test", client, True, tools=build_subagent_tools(), temperature=0)
        agent._run_tool = Mock(return_value='{"findings":"shared"}')
        return agent, client

    def test_subagents_run_once_and_final_fork_receives_shared_results(self):
        agent, client = self.make_agent()
        pair = agent.run_paired("query", ["/doc0", "/doc1"])
        self.assertEqual(client.chat.call_count, 1)
        self.assertEqual(agent._run_tool.call_count, 2)
        client.paired_final.assert_called_once()
        messages = client.paired_final.call_args.args[0]
        self.assertEqual(len([m for m in messages if m["role"] == "tool"]), 2)
        self.assertEqual(pair["shared_messages"], messages)
        self.assertIsNone(messages[1]["content"])

    def test_call_count_does_not_limit_execution(self):
        for count in (0, 1, 3):
            with self.subTest(count=count):
                agent, client = self.make_agent(count=count)
                agent.run_paired("query", ["/doc0", "/doc1"])
                self.assertEqual(agent._run_tool.call_count, count)
                client.paired_final.assert_called_once()

    def test_document_assignment_is_not_validated(self):
        agent, client = self.make_agent()
        agent.run_paired("query", ["/other", "/other"])
        self.assertEqual(agent._run_tool.call_count, 2)
        client.paired_final.assert_called_once()


class PairedSummaryTest(unittest.TestCase):
    def test_invalid_pairs_are_excluded_from_accuracy(self):
        from scripts.browsecomp.run_browsecomp import parse_args, run
        def row(valid, full, reuse):
            return {"valid_pair": valid, "branches": {
                "full_prefill": {"metrics": {"exact_match": full}},
                "kv_reuse": {"metrics": {"exact_match": reuse}}}}
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata.json"
            metadata.write_text(json.dumps([{"query_id": str(i)} for i in range(3)]))
            output = Path(tmp) / "results.jsonl"
            with patch("sys.argv", ["run", "--metadata", str(metadata), "--all", "--output", str(output)]):
                args = parse_args()
            with patch("scripts.browsecomp.run_browsecomp.run_one", side_effect=[
                    row(True, 1, 0), row(True, 1, 1), row(False, 0, 1)]), patch("builtins.print"):
                run(args)
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["num_completed"], 3)
            self.assertEqual(summary["num_valid_pairs"], 2)
            self.assertEqual(summary["num_invalid_pairs"], 1)
            self.assertEqual(summary["metrics"]["full_prefill"]["exact_match"], 1)
            self.assertEqual(summary["metrics"]["kv_reuse"]["exact_match"], 0.5)
            self.assertEqual(summary["paired_outcomes"], {"full_only_correct": 1, "both_correct": 1})


if __name__ == "__main__":
    unittest.main()
