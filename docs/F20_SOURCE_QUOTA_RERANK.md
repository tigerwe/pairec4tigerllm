# F20 来源配额重排

## 目标

在 DeepFM 对 50 个候选完成打分和稳定排序后，增加一个进程内、确定性的来源配额重排，
避免最终结果全部来自向量召回。该阶段不训练模型、不修改 DeepFM 分数，也不引入新的网络
服务。

```text
GenerativeRecall + MilvusRecall
  -> QuotaMultiRecall (50 candidates)
  -> DeepFM stable sort
  -> SourceQuotaTail rerank
  -> requested size
```

## 策略

配置名固定为 `source_quota_tail`：

```text
generative_selected = min(2, generative_input, requested_size)
vector_selected     = requested_size - generative_selected
output              = top vectors + top generative candidates
```

- 两个来源内部保持 DeepFM 顺序。
- 生成式商品连续放在输出末尾。
- DeepFM `item.Score` 和 `deepfm_score` 保持不变。
- 输出商品写入 `rerank_position` 和 `rerank_reason=source_quota_tail` 属性。
- 输入只允许 `generative_recall` 和 `milvus_recall`。
- 缺少生成式候选、未知来源、重复商品、候选数或输出数不满足契约时 fail-closed，返回
  `code=500`、`msg=rerank failed`、空 items。
- `size` 支持 `1..expected_candidates`；当前 expected candidates 固定为 50。

## 配置

默认配置不启用。隔离 `pairec-brpc-observed` 配置包含：

```json
{
  "RerankConfs": [{
    "name": "source_quota_tail",
    "enabled": true,
    "generative_source": "generative_recall",
    "vector_source": "milvus_recall",
    "expected_candidates": 50,
    "minimum_generative": 1,
    "max_generative": 2,
    "placement": "tail",
    "fail_closed": true
  }]
}
```

## 可观测性

请求级 `rerank` span 使用 `protocol=in_process`，并记录：

```text
policy, placement, input_count, output_count,
generative_input, vector_input,
generative_selected, vector_selected,
minimum_generative, maximum_generative, moved_count
```

低基数 Prometheus 指标：

```text
pairec_rerank_duration_seconds{policy,status}
pairec_rerank_requests_total{policy,status}
```

## 验证

本地：

```bash
go test -race -mod=vendor \
  ./services/rerank ./services/observability ./services/sort \
  ./vendor/github.com/alibaba/pairec/v2/web
go build -mod=vendor ./services/...
python3 -m unittest tests.test_pipeline_trace_summary
bash -n scripts/deploy_and_validate_pairec_brpc_observed.sh
python3 -m json.tool configs/pairec_config.brpc_observed.json >/dev/null
```

远端隔离验收：

```bash
REQUESTS=3 RUN_HTTP_AB=0 \
  bash scripts/deploy_and_validate_pairec_brpc_observed.sh

REQUESTS=1000 RUN_HTTP_AB=1 \
RERANK_MAX_P99_MS=1.0 CLIENT_MAX_P99_MS=122.622 \
  bash scripts/deploy_and_validate_pairec_brpc_observed.sh
```

正式门禁要求每个响应生成式数量为 1 或 2 且连续位于末尾；Trace 必须证明有两个生成式
输入时保留两个；rerank p99 不超过 1ms，客户端 E2E p99 不超过历史基线
`120.622ms + 2ms`。
