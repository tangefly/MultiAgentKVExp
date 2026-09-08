import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from lminfer.config import EngineConfig
from lminfer.engine import GenerationResult
from lminfer.server import create_app


class Tokenizer:
    chat_template = "test"
    def convert_tokens_to_ids(self, token):
        return 1
    def convert_ids_to_tokens(self, token):
        return "unknown"
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3, 4, 5, 6] if kwargs.get("tokenize") else "rendered prompt"


def result(grafted=0):
    return GenerationResult(request_id="test", prompt_tokens=6, output_tokens=[7],
        output_text='{"prediction":"test"}', finish_reason="stop", ttft_ms=1,
        decode_ms=1, kv_cache_bytes=0, reused_prompt_tokens=grafted,
        grafted_tokens=grafted, grafted_segments=int(grafted > 0))


class PairedHTTPTest(unittest.TestCase):
    def test_final_fork_does_not_write_back_session_kv(self):
        config = EngineConfig(model="test", max_model_len=100, tool_call_parser="none",
                              reuse_agent_kv_append=True)
        engine = SimpleNamespace(config=config, model=SimpleNamespace(config=SimpleNamespace()),
            tokenizer=Tokenizer(), model_name="test",
            generate=AsyncMock(side_effect=[result(), result(), result(2)]))
        with patch("lminfer.server.SessionKVStore") as cls:
            store = cls.return_value
            store.build_grafts.return_value = [SimpleNamespace(tokens=[3, 4])]
            store.propose.return_value = [object()]
            with TestClient(create_app(engine)) as client:
                body = {"model": "test", "messages": [{"role": "user", "content": "query"}],
                        "mode": "agent", "trace": ["main"], "temperature": 0, "max_tokens": 2}
                first = client.post("/v1/chat/completions", json=body)
                self.assertEqual(first.status_code, 200, first.text)
                body.update(session_id=first.json()["session_id"], trace=["main", "sub", "main"],
                            paired_final=True, tool_choice="none")
                store.reset_mock()
                final = client.post("/v1/chat/completions", json=body)
                self.assertEqual(final.status_code, 200, final.text)
                self.assertTrue(final.json()["valid_pair"])
                self.assertEqual(final.json()["object"], "paired.chat.completion")
                store.put.assert_not_called()
                store.clear_subs.assert_not_called()
                self.assertEqual(engine.generate.call_count, 3)
                body["stream"] = True
                self.assertEqual(client.post("/v1/chat/completions", json=body).status_code, 400)
                self.assertEqual(engine.generate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
