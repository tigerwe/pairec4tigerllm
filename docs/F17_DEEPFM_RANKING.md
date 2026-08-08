# F17 DeepFM 精排工程闭环

## 链路

实验链路保持现有多路召回行为不变：

```text
PaiRec multi recall (Generative + Milvus, 50 candidates)
  -> DeepFMRankSort
  -> http://141.61.91.189:18210/rank
  -> stable sort by DeepFM score
  -> truncate to requested size
```

`pairec-multi-recall-rank` 是独立 Deployment 和 ClusterIP Service。已验收的
`pairec-multi-recall` 不会被修改。

## 模型契约

- PyTorch DeepFM，EasyRec DeepFM 仅作为结构参考。
- 输入字段：`user_id`、`item_id`、`video_category`、`gender`、`age`、
  `hist_1` 到 `hist_10`。
- 标签：`click`。
- 词表与 DSSM 共用，索引 0 为 OOV/PAD。
- 数据按固定 seed 做确定性 90/10 train/validation 切分。
- 最多 10 epochs，validation logloss early stopping，patience=2。
- AUC 与 logloss 只记录，不作为质量门禁。

当前 `deepfm_out` 基于旧的部分 DSSM 词表训练，只用于验证接口、部署、时延和失败
语义，启动时必须声明 `MODEL_ROLE=engineering`。DSSM 全量词表冻结后重训得到的模型才可
声明为 `production_candidate`；该角色会自动启用真实推荐请求的零 OOV 门禁。

训练产物固定为：

```text
deepfm_best.pt
feature_vocab.json
model_config.json
training_summary.json
item_categories.json
user_profiles.json
```

## 在线契约

Rank Service 只打分，不排序。每次请求必须包含 50 个唯一 item ID，响应必须返回
完全相同的 ID 集合、有限分数、同一 request ID 和非空 model version。PaiRec 按分数
降序稳定排序，同分保持召回顺序。

服务同时返回 `model_role`、checkpoint epoch、词表规模以及 user/item/category/profile/
history OOV 计数。启动时会核对 checkpoint 中的 vocab sizes 与挂载的
`feature_vocab.json`，混用模型与词表会直接失败。协议验收从模型词表选择真实用户和
50 个真实商品，要求所有特征命中且至少产生两个不同分数，不再使用可能全 OOV 的
固定 `1..50` 商品。

Rank Service 超时、不可用、返回错误或候选集合不一致时，PaiRec 不回退召回顺序：
HTTP 状态保持 200，业务体返回 `code=500`、`msg=deepfm rank failed`、空 items。
实验 Pod 的 readiness 同时依赖 Rank Service 健康状态。

## 执行

训练：

```bash
CSV_PATH=/path/to/ctr_data_1M.csv \
DSSM_VOCAB_PATH=/path/to/dssm_out/vocab.json \
OUTPUT_DIR=/path/to/deepfm_out \
  bash scripts/run_deepfm_train.sh
```

worker1 使用全量词表的推荐入口：

```bash
REPO_DIR=/home/zcx/workspace/pairec4tigerllm \
OUTPUT_DIR=/home/zcx/workspace/pairec4tigerllm/deepfm_full_vocab_out \
  bash scripts/run_deepfm_full_vocab_worker1.sh \
  | tee /tmp/deepfm-full-vocab.log
```

在 master 启动独立 CPU Rank Service：

```bash
REPO_DIR=/home/zcx/workspace/pairec4tigerllm \
MODEL_DIR=/home/zcx/workspace/pairec4tigerllm/deepfm_out \
MODEL_ROLE=engineering \
  bash scripts/run_deepfm_rank_container.sh
```

部署并验收实验链路：

```bash
DEEPFM_MODEL_DIR=/home/zcx/workspace/pairec4tigerllm/deepfm_out \
DEEPFM_MODEL_ROLE=engineering \
  bash scripts/deploy_and_validate_pairec_deepfm_rank.sh | \
  tee /tmp/pairec-deepfm-rank-validation.log
```

脚本依次执行 Rank 协议 100 次验证、50/10 两组 smoke、size=10 的 100 次稳定性，
并默认停止 Rank 容器做 fail-closed 与 NotReady 故障注入，最后自动恢复服务。工程模型
会汇总真实候选 OOV，但不以 OOV 阻断链路；`production_candidate` 默认要求所有请求的
user/item/category/gender/age/history OOV 均为零，也可显式用
`REQUIRE_ZERO_RANK_OOV=1` 提前执行同一门禁。
