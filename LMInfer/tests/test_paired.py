import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from lminfer.config import EngineConfig, SamplingParams
from lminfer.engine import GenerationResult, LLMEngine
from lminfer.kvcache import KVGraft, KVPrefix, tail_cache
from lminfer.paired import generate_pair


def result(reused=0, grafted=0, mismatch=False):
    return GenerationResult(request_id="test", prompt_tokens=6, output_tokens=[8],
        output_text='{"prediction":"test"}', finish_reason="stop", ttft_ms=1,
        decode_ms=2, kv_cache_bytes=0, reused_prompt_tokens=reused,
        grafted_tokens=grafted, grafted_segments=int(grafted > 0), kv_graft_mismatch=mismatch)


class PairTest(unittest.IsolatedAsyncioTestCase):
    async def test_identical_prompt_and_only_reuse_branch_has_sources(self):
        engine = SimpleNamespace(config=SimpleNamespace(max_model_len=100),
                                 generate=AsyncMock(side_effect=[result(), result(3, 2)]))
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
        sources = [object()]
        graft = [SimpleNamespace(tokens=[3, 4])]
        pair = await generate_pair(engine, "p", prompt, SamplingParams(temperature=0, max_tokens=2), sources, graft)
        full, reuse = engine.generate.call_args_list
        self.assertTrue(torch.equal(full.args[1], reuse.args[1]))
        self.assertNotEqual(full.args[1].data_ptr(), reuse.args[1].data_ptr())
        self.assertIsNone(full.kwargs["reuse_prefixes"])
        self.assertIsNone(full.kwargs["graft"])
        self.assertIs(reuse.kwargs["reuse_prefixes"], sources)
        self.assertIs(reuse.kwargs["graft"], graft)
        self.assertTrue(pair["valid_pair"])
        self.assertEqual(pair["prompt_token_ids"], prompt[0].tolist())

    async def test_rejects_missing_graft_sampling_and_truncation(self):
        engine = SimpleNamespace(config=SimpleNamespace(max_model_len=8), generate=AsyncMock())
        for sampling, graft in [(SamplingParams(temperature=0, max_tokens=2), []),
                                 (SamplingParams(temperature=0.3, max_tokens=2), [object()]),
                                 (SamplingParams(temperature=0, max_tokens=3), [object()])]:
            with self.assertRaises(ValueError):
                await generate_pair(engine, "p", torch.ones((1, 6), dtype=torch.long), sampling, [], graft)
        engine.generate.assert_not_called()

    async def test_fallback_is_not_a_valid_experiment(self):
        engine = SimpleNamespace(config=SimpleNamespace(max_model_len=100),
                                 generate=AsyncMock(side_effect=[result(), result(2, 0, True)]))
        pair = await generate_pair(engine, "p", torch.ones((1, 6), dtype=torch.long),
            SamplingParams(temperature=0, max_tokens=2), [], [SimpleNamespace(tokens=[1])])
        self.assertFalse(pair["valid_pair"])

    async def test_real_cpu_engine_graft_and_source_cache_isolation(self):
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=128, eos_token_id=None)
        config._attn_implementation = "eager"
        engine = LLMEngine.__new__(LLMEngine)
        engine.config = EngineConfig(model="tiny-local-test", max_model_len=128, disable_log_stats=True)
        engine.model = Qwen3ForCausalLM(config).eval()
        engine.tokenizer = SimpleNamespace(decode=lambda ids, **kw: " ".join(map(str, ids)))
        engine._think_ids = None
        engine._stats = {"completed": 0, "generated_tokens": 0, "prefill_tokens": 0}
        engine.executor = ThreadPoolExecutor(max_workers=1)
        try:
            with torch.inference_mode():
                main_cache = engine.model(torch.tensor([[1, 2]]), use_cache=True).past_key_values
                sub_cache = engine.model(torch.tensor([[9, 10, 3, 4]]), use_cache=True).past_key_values
            prefix = KVPrefix(tokens=[1, 2], cache=main_cache)
            graft = KVGraft(position=2, tokens=[3, 4],
                           cache=tail_cache(sub_cache, 2, config), source_position=2)
            before = [(layer.keys.clone(), layer.values.clone()) for layer in graft.cache.layers]
            main_before = [(layer.keys.clone(), layer.values.clone()) for layer in main_cache.layers]
            pair = await generate_pair(engine, "real", torch.tensor([[1, 2, 3, 4, 5, 6]]),
                SamplingParams(temperature=0, max_tokens=3), [prefix], [graft])
            self.assertTrue(pair["valid_pair"])
            self.assertEqual(pair["branches"]["full_prefill"]["reused_prompt_tokens"], 0)
            self.assertEqual(pair["branches"]["kv_reuse"]["grafted_tokens"], 2)
            for cache, snapshot in [(graft.cache, before), (main_cache, main_before)]:
                for layer, (keys, values) in zip(cache.layers, snapshot):
                    self.assertTrue(torch.equal(layer.keys, keys))
                    self.assertTrue(torch.equal(layer.values, values))
        finally:
            engine.executor.shutdown()


if __name__ == "__main__":
    unittest.main()
