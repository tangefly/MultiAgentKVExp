"""Final-answer fork. Neither branch writes back to the session KV store."""
import hashlib
import json

from .toolcalls import clean_content


async def generate_pair(engine, request_id, prompt_ids, sampling, reuse_prefixes, graft):
    if not sampling.greedy:
        raise ValueError("paired_final requires temperature=0")
    if prompt_ids.shape[1] + sampling.max_tokens > engine.config.max_model_len:
        raise ValueError("paired_final refuses prompt truncation; increase --max-model-len or reduce max_tokens")
    if not graft:
        raise ValueError("No SubAgent KV matched the final prompt; cannot run a valid reuse experiment")
    tokens = prompt_ids[0].tolist()
    digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    branches = {}
    # Run sequentially, freeing the full branch cache before running the reuse branch.
    # The engine's slice_cache/concat_cache clone source KV before mutable decode.
    for name in ("full_prefill", "kv_reuse"):
        reuse = name == "kv_reuse"
        result = await engine.generate(
            request_id + "-" + name, prompt_ids.clone(), sampling,
            skip_special_tokens=False,
            reuse_prefixes=reuse_prefixes if reuse else None,
            graft=graft if reuse else None,
        )
        branches[name] = {
            "raw_answer": result.output_text,
            "answer": clean_content(result.output_text) or "",
            "output_token_ids": result.output_tokens,
            "finish_reason": result.finish_reason,
            "prompt_sha256": digest,
            "usage": {"prompt_tokens": result.prompt_tokens,
                      "completion_tokens": result.completion_tokens,
                      "total_tokens": result.prompt_tokens + result.completion_tokens},
            "reused_prompt_tokens": result.reused_prompt_tokens,
            "grafted_tokens": result.grafted_tokens,
            "grafted_segments": result.grafted_segments,
            "kv_graft_mismatch": result.kv_graft_mismatch,
            "ttft_ms": result.ttft_ms,
            "decode_ms": result.decode_ms,
        }
        result.kv_cache = None
        if result.prompt_tokens != len(tokens):
            raise RuntimeError("Engine changed the paired prompt length")
        if not reuse and result.reused_prompt_tokens != 0:
            raise RuntimeError("Full prefill unexpectedly reused KV")
    reuse_result = branches["kv_reuse"]
    valid = reuse_result["grafted_tokens"] > 0 and not reuse_result["kv_graft_mismatch"]
    return {
        "valid_pair": valid,
        "invalid_reason": None if valid else "No actual SubAgent graft, or graft validation fallback",
        "prompt_token_ids": tokens,
        "prompt_sha256": digest,
        "branch_order": ["full_prefill", "kv_reuse"],
        "planned_graft_segments": len(graft),
        "planned_graft_tokens": sum(len(g.tokens) for g in graft),
        "branches": branches,
    }
