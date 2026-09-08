# MultiAgentKVExp：共享 SubAgent 轨迹的配对实验

本目录的 BenchAgent 和 LMInfer 是独立实验副本。不要使用原项目启动的服务，也不需要 vLLM。

## 运行

在装有 LMInfer 依赖的 Python 环境中，从本目录运行。将 `/path/to/Qwen3-8B` 换成模型目录。

```bash
cd /home/tanger/workspace/MultiAgentKVExp
PYTHONPATH="$PWD/LMInfer" python3 -m lminfer.cli serve /path/to/Qwen3-8B \
  --served-model-name Qwen3-8B --port 8001 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reuse-agent-kv-append --graft-rope-rebase --max-num-seqs 1
```

另一个终端运行：

```bash
python3 /home/tanger/workspace/MultiAgentKVExp/BenchAgent/scripts/browsecomp/run_browsecomp.py \
  --base-url http://localhost:8001/v1 --model Qwen3-8B --index 0 \
  --temperature 0 --enable-thinking --sub-max-tokens 10240 \
  --sub-max-iters 4 --final-max-tokens 10240 --no-continue-on-error
```

批量使用 `--limit 100` 或 `--all` 替换 `--index 0`。服务必须允许 prompt + max_tokens 的长度；
配对请求拒绝截断输入。默认每个样本结束释放会话 KV，可使用 `--no-release-kv` 保留。
`PYTHONPATH` 显式指向本副本，避免原目录的 editable installation 被误用。

## 实验流程

1. MainAgent 只规划一次。提示词建议按文档分配 SubAgent，但客户端直接执行返回的全部工具调用，
   不校验调用数量与文档数量是否一致，也不校验 document_index/document_path。
2. 每个 SubAgent 运行一次完整工具循环，默认串行执行。它们与 MainAgent 共用服务端 session；
   LMInfer 保留各 SubAgent 最后一次生成的 token 与 KV。共享阶段可以按服务器配置复用前缀。
3. 所有结果返回后，客户端调用同一个 chat 接口，携带 `paired_final=true`、`tool_choice=none`。
   两个最终分支都不再调用工具。最终 prompt 不渲染可用工具定义，保留完整工具调用历史和结果。
4. 服务端只构建一份 prompt token IDs，先执行 full_prefill，再执行 kv_reuse：
   - full_prefill：不传任何跨请求前缀或 graft；decode 仍正常使用本次新计算的 KV。
   - kv_reuse：使用 Main 历史的可匹配前缀，以及 SubAgent 输出的可匹配连续 KV 片段。
   引擎深拷贝源缓存后才进行可变计算；两路最终生成均不写回 session KV。
5. 没有可匹配 SubAgent KV，或实际 graft 为零时，释放该次会话 KV 并重跑整个样本，
   最多额外重试 2 次（共 3 次，每次新会话）。仍失败则保存 skipped/skip_reason/attempts，
   继续下一个样本，即使设置了 --no-continue-on-error。其他错误沿用 continue-on-error 策略。
   summary 用 num_skipped 单独计数；跳过样本不计入准确度。温度为 0 时重跑也可能重复失败。

不是所有正文 token 都必然拼接：模板边界、最长连续匹配和 repair 参数可能要求部分重新计算。
`planned_graft_*` 表示候选量，分支内的 `grafted_*` 表示实际量；
`reused_prompt_tokens` 还包含历史前缀，不等于 SubAgent 拼接量。
RoPE rebase 只修正位置，不消除源上下文与 Main 上下文的差异。

## 结果

JSONL 默认写入 `BenchAgent/outputs/browsecomp/`，每个样本包含：

- `index`、`query_id`、`query`、`gold`：样本和标准答案；
- `branches.full_prefill` / `branches.kv_reuse`：清理后的完整答案（含 support）、prediction、
  指标、finish_reason、usage、实际复用/拼接量、回退标记、TTFT 和 decode 时间；
- `valid_pair`、`invalid_reason`、`same_prediction`、`exact_match_delta`：配对状态与差异。

JSONL 不保存 sampling、配置快照、完整 token 列表、共享消息历史、原始计划、thinking
或重复的原始答案。此次精简只影响新运行的结果文件。

summary 分别统计两组 ROUGE、token F1、exact match，以及 both_correct / reuse_only_correct /
full_only_correct / both_wrong。准确度只评价 prediction，support 和证据忠实度需另行审阅。
无效配对和错误单独计数。`finish_reason=length` 也保留，便于检查输出被截断的情况。
结果保留运行中的时间观测，但分支固定顺序、设备预热和源缓存复制会影响时间；
当前实验主要用于质量比较，TTFT 沿用引擎统计，不包括所有服务端准备开销。

## 测试

```bash
cd /home/tanger/workspace/MultiAgentKVExp/LMInfer
python3 -m unittest discover -s tests -v
cd /home/tanger/workspace/MultiAgentKVExp/BenchAgent
python3 -m unittest discover -s tests -v
```

包含 CPU 随机初始化的微型 Qwen3 实测 KV 拼接/源缓存隔离、同输入分叉、回退/截断保护、
HTTP 配对请求不覆盖缓存，以及 MainAgent 只执行一次共享任务的测试。不下载模型。
