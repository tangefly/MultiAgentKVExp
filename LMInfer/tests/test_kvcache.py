import unittest

import torch
from transformers import DynamicCache

from lminfer.kvcache import (
    KIND_MAIN,
    KIND_SUB,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    SessionKVStore,
)


class FakeTokenizer:
    ids = {TOOL_RESPONSE_OPEN: 900, TOOL_RESPONSE_CLOSE: 901}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def convert_ids_to_tokens(self, token_id):
        for token, tid in self.ids.items():
            if tid == token_id:
                return token
        return f"tok_{token_id}"


def make_cache(length: int) -> DynamicCache:
    keys = torch.arange(length * 2, dtype=torch.float32).reshape(1, 1, length, 2)
    values = keys + 1000
    return DynamicCache(ddp_cache_data=[(keys, values)], config=None)


class SessionKVStoreTest(unittest.TestCase):
    def test_build_grafts_reuses_all_subs_since_latest_main(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3, 4]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(4), prompt_len=4))

        sub_outputs = [
            [101, 102, 103, 104],
            [201, 202, 203, 204],
            [301, 302, 303, 304],
        ]
        for i, output in enumerate(sub_outputs, start=1):
            seq = [10 * i, 10 * i + 1] + output
            self.assertTrue(
                store.put(session_id, KIND_SUB, seq, make_cache(len(seq)), prompt_len=2)
            )

        prompt = (
            main_tokens
            + [900] + sub_outputs[0] + [901, 88]
            + [900] + sub_outputs[1] + [901, 89]
            + [900] + sub_outputs[2] + [901, 77]
        )
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub3", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], sub_outputs)
        self.assertEqual([g.position for g in grafts], [5, 12, 19])
        self.assertEqual([g.cache.get_seq_length() for g in grafts], [4, 4, 4])
        self.assertEqual(len(store.propose(session_id, [KIND_MAIN])), 4)



    def test_build_grafts_partially_matches_one_tool_response_window(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))

        sub_out = [101, 102, 103, 104, 201, 202, 203, 204]
        sub_seq = [9] + sub_out
        self.assertTrue(store.put(session_id, KIND_SUB, sub_seq, make_cache(len(sub_seq)), prompt_len=1))

        # 900/901 are tool_response markers. The window has unmatched boundary tokens
        # and an unmatched token in the middle. A sub invocation is treated as one
        # KV segment, so only one longest contiguous span is grafted.
        prompt = main_tokens + [900, 77, 101, 102, 103, 104, 88, 201, 202, 203, 204, 99, 901]
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], [[101, 102, 103, 104]])
        self.assertEqual([g.position for g in grafts], [5])
        self.assertEqual([g.source_position for g in grafts], [1])


    def test_sub_put_replaces_same_trace_invocation(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2], make_cache(2), prompt_len=2,
                                  trace=[KIND_MAIN]))

        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 9, 10], make_cache(4),
                                  prompt_len=1, trace=[KIND_MAIN, KIND_SUB]))
        self.assertTrue(store.put(session_id, KIND_SUB, [11, 12, 13, 14, 15], make_cache(5),
                                  prompt_len=1, trace=[KIND_MAIN, KIND_SUB]))
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[1].tokens, [11, 12, 13, 14, 15])

        self.assertTrue(store.put(session_id, KIND_SUB, [21, 22, 23, 24], make_cache(4),
                                  prompt_len=1, trace=[KIND_MAIN, KIND_SUB, KIND_SUB]))
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 3)
        self.assertEqual([c.tokens for c in candidates[1:]],
                         [[11, 12, 13, 14, 15], [21, 22, 23, 24]])

    def test_build_grafts_skips_sub_internal_intermediate_outputs(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3, 4]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(4), prompt_len=4))

        final_outputs = [
            [101, 102, 103, 104],
            [201, 202, 203, 204],
            [301, 302, 303, 304],
        ]
        internal_outputs = [
            [11, 12, 13, 14, 15, 16],
            [21, 22, 23, 24, 25, 26],
            [31, 32, 33, 34, 35, 36],
        ]
        for i, output in enumerate(final_outputs):
            internal_seq = [40 + i] + internal_outputs[i]
            final_seq = [50 + i, 60 + i] + output
            self.assertTrue(
                store.put(session_id, KIND_SUB, internal_seq, make_cache(len(internal_seq)), prompt_len=1)
            )
            self.assertTrue(
                store.put(session_id, KIND_SUB, final_seq, make_cache(len(final_seq)), prompt_len=2)
            )

        prompt = (
            main_tokens
            + [900] + final_outputs[0] + [901, 88]
            + [900] + final_outputs[1] + [901, 89]
            + [900] + final_outputs[2] + [901, 77]
        )
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", "sub", "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], final_outputs)
        self.assertEqual([g.position for g in grafts], [5, 12, 19])


    def test_build_grafts_uses_main_lcp_and_skips_unmatched_windows(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        # Saved main output is longer than the structured assistant tool-call rendering
        # in the final prompt, so len(main_seg.tokens) would skip the first real window.
        main_saved = [1, 2, 3, 70, 71, 72, 73, 74, 75, 76]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_saved, make_cache(len(main_saved)), prompt_len=3))

        outputs = [[101, 102, 103, 104], [201, 202, 203, 204]]
        for i, output in enumerate(outputs):
            # Add one intermediate sub output before each final answer. These should be skipped.
            self.assertTrue(store.put(session_id, KIND_SUB, [10 + i, 41, 42, 43, 44], make_cache(5), prompt_len=1))
            self.assertTrue(store.put(session_id, KIND_SUB, [20 + i] + output, make_cache(5), prompt_len=1))

        prompt = (
            [1, 2, 3, 80, 81]
            + [900, 51, 52, 53, 54, 901]  # stale/unmatched window before real tool results
            + [900] + outputs[0] + [901, 88]
            + [900] + outputs[1] + [901, 89]
        )
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], outputs)
        self.assertEqual([g.position for g in grafts], [12, 19])


    def test_build_grafts_prefers_final_answer_over_short_spurious_match(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))

        # The intermediate sub turn shares a short phrase with the returned tool response,
        # but the later final sub answer covers much more and must be selected.
        intermediate = [10, 101, 102, 103, 104, 99]
        final = [201, 202, 101, 102, 103, 104, 105, 106, 107, 108, 203]
        self.assertTrue(store.put(session_id, KIND_SUB, [7] + intermediate, make_cache(7), prompt_len=1))
        self.assertTrue(store.put(session_id, KIND_SUB, [8] + final, make_cache(12), prompt_len=1))

        prompt = main_tokens + [900, 77, 101, 102, 103, 104, 105, 106, 107, 108, 88, 901]
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], [[101, 102, 103, 104, 105, 106, 107, 108]])
        self.assertEqual([g.position for g in grafts], [5])
        self.assertEqual([g.source_position for g in grafts], [3])

    def test_build_grafts_excludes_leading_think_block(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))

        answer = [101, 102, 103, 104, 105, 106]
        # 输出以 <think> 块开头, think 内复述了完整答案(Qwen3 常见);
        # 剔除 think 后 graft 必须落在真正的答案区域, 而不是 think 内的复述。
        think = [201, 202] + answer + [203]
        sub_seq = [7, 8] + think + answer
        self.assertTrue(store.put(session_id, KIND_SUB, sub_seq, make_cache(len(sub_seq)),
                                  prompt_len=2, think_len=len(think)))

        prompt = main_tokens + [900] + answer + [901]
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], [answer])
        self.assertEqual([g.position for g in grafts], [4])
        # 源位置落在 think 块之后的真正输出区域
        self.assertEqual([g.source_position for g in grafts], [2 + len(think)])

    def test_build_grafts_anchors_document_order_within_boundary_drift(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))

        # 窗口 1 的正文是 sub 1 的答案, 但 sub 1 首 token 因边界 BPE 合并丢失,
        # 真实匹配只有 6 tok; sub 2 的输出恰好包含窗口 1 的完整正文(两个 sub
        # 答案高度重合), 跨 sub 匹配达 7 tok。文档序锚定必须仍选 sub 1。
        window1_body = [21, 22, 23, 24, 25, 26, 27]
        sub1_out = [20, 22, 23, 24, 25, 26, 27]      # 首 token 漂移
        sub2_out = [21, 22, 23, 24, 25, 26, 27, 28]  # 更长, 会被全局最长匹配选中
        for out in (sub1_out, sub2_out):
            seq = [9, 10] + out
            self.assertTrue(store.put(session_id, KIND_SUB, seq, make_cache(len(seq)), prompt_len=2))

        prompt = (main_tokens
                  + [900] + window1_body + [901, 88]
                  + [900] + sub2_out + [901, 89])
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", "sub", KIND_MAIN], prompt)

        self.assertEqual(len(grafts), 2)
        self.assertEqual(grafts[0].tokens, [22, 23, 24, 25, 26, 27])
        self.assertEqual(grafts[0].source_position, 3)  # sub 1 输出区第 2 个 token
        self.assertEqual(grafts[1].tokens, sub2_out)
        self.assertEqual(grafts[1].source_position, 2)  # sub 2 输出区起点

    def test_build_grafts_rejects_short_cross_sub_boilerplate_match(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))

        # 窗口正文属于未保存 KV 的 sub(如请求被截断); 剩余 sub 只与窗口共享
        # 4 个样板 token, 达不到覆盖率门槛, 不得误拼。
        body = list(range(300, 320))  # 20 tok
        other_out = [21, 22] + [301, 302, 303, 304] + [23, 24]
        self.assertTrue(store.put(session_id, KIND_SUB, [9, 10] + other_out,
                                  make_cache(10), prompt_len=2))

        prompt = main_tokens + [900] + body + [901]
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertEqual(grafts, [])

    def test_new_main_clears_previous_sub_batch(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2], make_cache(2), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 9, 10, 11, 12], make_cache(6), prompt_len=2))
        self.assertEqual(len(store.propose(session_id, [KIND_MAIN])), 2)

        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2, 3], make_cache(3), prompt_len=3))
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tokens, [1, 2, 3])
        self.assertEqual(store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], [1, 2, 3]), [])

    def test_clear_subs_releases_only_sub_batch(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2], make_cache(2), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [3, 4, 5, 6], make_cache(4), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 9, 10], make_cache(4), prompt_len=2))

        store.clear_subs(session_id)
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tokens, [1, 2])

    def test_build_graft_compat_returns_last_matched_sub(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1], make_cache(1), prompt_len=1))
        self.assertTrue(store.put(session_id, KIND_SUB, [5, 6, 21, 22, 23, 24], make_cache(6), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 31, 32, 33, 34], make_cache(6), prompt_len=2))

        prompt = [1, 900, 21, 22, 23, 24, 901, 900, 31, 32, 33, 34, 901]
        graft = store.build_graft(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertIsNotNone(graft)
        self.assertEqual(graft.tokens, [31, 32, 33, 34])
        self.assertEqual(graft.position, 8)


if __name__ == "__main__":
    unittest.main()
