# 数据分析 Agent MVP 工程方案

## 1. 开工前输出

### 1.1 模块清单

#### `src/agent/`

| 模块 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `state.py` | 定义 `AgentState`、`StageCard`、`RunContext` 等状态模型（只存 key/摘要） | 各 Node 运行结果摘要 | 可序列化状态对象 |
| `data_manager.py` | DataFrame 生命周期管理（注册、读取、版本、释放） | 原始 DF / `df_key` | 新 `df_key` / DF 引用（仅 Tool 内） |
| `decision_rules.py` | 非 LLM 的硬规则与候选策略生成（任务/结构识别、Mixed Effect 触发） | profile、schema、用户意图 | candidates、hard constraints |
| `observer.py` | Streaming 日志、Stage 卡片、事件回调 | Node 事件与阶段结果 | `stream.log`、`stage_card.json` |
| `graph.py` | LangGraph 工作流编排与条件路由 | `AgentState` | 更新后的 `AgentState` + artifacts 索引 |

#### `src/skills/profiling/`

| 模块 | 职责 |
|---|---|
| `dataset_profile.py` | 输出列类型、缺失、分布、唯一值、时间跨度、组结构等 profile |

#### `src/skills/eda/`

| 模块 | 职责 |
|---|---|
| `basic_stats.py` | 数值/类别统计、相关性、描述性分析 |
| `visualization.py` | 自动图表（分布、箱线、相关热图、时间趋势）并落盘 |

#### `src/skills/cleaning/`

| 模块 | 职责 |
|---|---|
| `missing.py` | 缺失值策略（删除/插补） |
| `outlier.py` | 异常值检测与处理 |
| `type_fix.py` | 类型标准化（时间、类别、数值） |
| `dedup.py` | 去重与一致性修复 |

#### `src/skills/feature_engineering/`

| 模块 | 职责 |
|---|---|
| `encoding.py` | 类别编码（OneHot/Target 编码候选） |
| `scaling.py` | 标准化/归一化 |
| `feature_select.py` | 特征筛选（方差、相关性、模型重要度） |
| `time_features.py` | 时间派生特征（lag/rolling/cycle） |
| `interaction.py` | 交互项与非线性特征（按计划启用） |

#### `src/skills/modeling/`

| 模块 | 职责 |
|---|---|
| `baseline.py` | 回归/分类/时序基础模型训练与预测 |
| `mixed_effect.py` | 强制场景下 MixedLM（随机截距 + 固定效应）与 OLS 对比 |
| `evaluation.py` | 任务相关指标、交叉验证、误差分析与可视化 |

#### `src/skills/reporting/`

| 模块 | 职责 |
|---|---|
| `artifacts.py` | 统一 artifact 落盘、索引、读取摘要 |

#### `src/app/`

| 模块 | 职责 |
|---|---|
| `config_loader.py` | `.env` 与 YAML 配置加载、校验 |
| `local_llm.py` | 公司本地 LLM API 封装（Planner/Interpreter/Analyst 公共调用层） |
| `mock_llm.py` | 离线或测试桩，输出稳定 JSON |
| `agent_runner.py` | CLI/服务入口，驱动图执行 |
| `streamlit_app.py` | 交互界面，展示日志、卡片、最终报告、追问 |

#### `tests/`

| 测试文件 | 覆盖重点 |
|---|---|
| `test_data_manager.py` | `df_key` 映射、隔离、生命周期 |
| `test_decision_rules.py` | 任务识别、Mixed Effect 触发规则 |
| `test_profiling.py` | profile 输出稳定性与关键字段 |
| `test_mixed_effect.py` | MixedLM 产物完整性、ICC 与 OLS 对比 |
| `test_pipeline_smoke.py` | 端到端冒烟（含 logs + stage cards + summary） |

---

### 1.2 接口签名（建议落地版）

```python
# src/agent/state.py
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

TaskType = Literal["regression", "classification", "timeseries", "survival", "eda"]
DataStructure = Literal["hierarchical", "repeated_measure", "high_d_low_n", "flat_iid"]
StageName = Literal["profiling", "cleaning", "feature_engineering", "modeling", "evaluation"]


class ArtifactRef(BaseModel):
    name: str
    path: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class StageCard(BaseModel):
    stage: StageName
    key_findings: List[str]
    artifacts: List[ArtifactRef]
    next_step: str


class PlannerOutput(BaseModel):
    task_type: TaskType
    data_structure: DataStructure
    target_column: Optional[str] = None
    only_eda: bool = False
    use_mixed_effect: bool = False
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    feature_strategy: Dict[str, Any] = Field(default_factory=dict)
    evaluation_strategy: Dict[str, Any] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    run_id: str
    df_key: str
    user_intent: str
    schema_path: Optional[str] = None
    profile_summary: Dict[str, Any] = Field(default_factory=dict)
    plan: Optional[PlannerOutput] = None
    stage_cards: List[StageCard] = Field(default_factory=list)
    artifact_index: Dict[str, str] = Field(default_factory=dict)
    stream_events: List[Dict[str, Any]] = Field(default_factory=list)
    final_summary: Optional[str] = None
```

```python
# src/agent/data_manager.py
import pandas as pd
from typing import Optional


class DataManager:
    def register(self, df: pd.DataFrame, source_name: str) -> str: ...
    def get(self, df_key: str) -> pd.DataFrame: ...
    def clone(self, df_key: str, new_name: str) -> str: ...
    def put(self, df_key: str, df: pd.DataFrame) -> None: ...
    def release(self, df_key: str) -> None: ...
```

```python
# src/agent/decision_rules.py
from typing import Any, Dict, List


def infer_task_candidates(profile: Dict[str, Any], schema: Dict[str, Any], user_intent: str) -> List[Dict[str, Any]]: ...
def infer_data_structure(profile: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]: ...
def must_use_mixed_effect(profile: Dict[str, Any], schema: Dict[str, Any]) -> bool: ...
def build_planner_candidates(profile: Dict[str, Any], schema: Dict[str, Any], user_intent: str) -> Dict[str, Any]: ...
```

```python
# src/app/local_llm.py
from typing import Any, Dict, List


class LocalLLMClient:
    def plan(self, profile: Dict[str, Any], schema: Dict[str, Any], candidates: Dict[str, Any], user_intent: str) -> Dict[str, Any]: ...
    def interpret_stage(self, stage: str, stage_result: Dict[str, Any]) -> Dict[str, Any]: ...
    def answer(self, question: str, artifact_summaries: List[Dict[str, Any]], final_summary: str) -> str: ...
```

```python
# src/skills/reporting/artifacts.py
from typing import Any, Dict


def save_json(run_id: str, stage: str, name: str, payload: Dict[str, Any]) -> str: ...
def save_csv(run_id: str, stage: str, name: str, df) -> str: ...
def save_plot(run_id: str, stage: str, name: str, fig) -> str: ...
def upsert_artifact_index(run_id: str, key: str, path: str) -> None: ...
```

```python
# src/agent/observer.py
from typing import Any, Dict, List


class Observer:
    def on_node_start(self, run_id: str, node: str, df_key: str) -> None: ...
    def on_node_end(self, run_id: str, node: str, outputs: Dict[str, Any]) -> None: ...
    def emit_stage_card(self, run_id: str, card: Dict[str, Any]) -> None: ...
    def emit_stream_event(self, run_id: str, event: Dict[str, Any]) -> None: ...
```

```python
# src/agent/graph.py
from .state import AgentState


def build_graph(use_mock_llm: bool = False): ...
def run_pipeline(initial_state: AgentState) -> AgentState: ...
```

---

### 1.3 run_id 规范

- 格式：`run_YYYYMMDD_HHMMSS_<8hex>`
- 示例：`run_20260212_103045_a1b2c3d4`
- 生成时机：每次新任务启动时生成一次，全链路透传，不可复用。
- 作用范围：
  - 状态追踪：`AgentState.run_id`
  - 落盘目录：`storage/runs/{run_id}/...`
  - 日志关联：`stream.log`、`events.jsonl`
  - 前端会话绑定：用于追问定位当前 artifacts
- 冲突策略：若目录已存在，在末尾追加 `_r{n}`（极低概率兜底）。

---

## 2. 需求理解与边界确认

### 2.1 目标（MVP 必达）

1. 给定 CSV/Excel 自动识别任务类型与数据结构。
2. 读取 `column_schema.json` + profile + 用户意图，由 LLM Planner 生成 `plan.json`。
3. LangGraph 按计划执行五阶段：Profiling -> Cleaning -> Feature Engineering -> Modeling -> Evaluation。
4. 全程产出可视化、可解释卡片、可追问问答、可落盘 artifacts。
5. 若判定层级/重复测量数据，必须执行 Mixed Effect 并输出规定文件。

### 2.2 绝对约束（架构红线）

1. 不允许公网搜索、外部在线 AutoML、外部工具 API。
2. DataFrame 不能进入 `AgentState`；只能通过 `df_key` 在 `DataManager` 内部流转。
3. 传给 LLM 的仅限摘要（profile/schema/stage_result），禁止全量数据。
4. 代码中不写死厂商 API 地址，统一由 `.env` 注入。
5. 必须有 Streaming Logs 与 Stage 卡片，否则验收失败。

### 2.3 非目标（MVP 不做）

1. 分布式训练与超大数据集优化。
2. 自动调参平台化（仅支持轻量候选策略）。
3. 多租户权限体系（保留接口，后续迭代）。

---

## 3. 系统架构设计

### 3.1 四层职责映射

1. Tool 层：原子、无策略、输入 `df_key` + 参数，输出 artifacts 路径 + JSON。
2. Skill 层：阶段单元，组合 Tool，提供确定性流水线步骤。
3. Graph 层：状态机与路由，管理阶段顺序、条件分支、重试与事件。
4. LLM 层：只做决策与解释（Planner/Interpreter/Analyst），不直接处理原始 DF。

### 3.2 执行主流程

1. 读入数据 -> `DataManager.register` -> 得到 `df_key`。
2. Profiling Skill 生成 `profile.json` + EDA 预览图。
3. Decision Rules 产出候选策略 + 硬约束（例如是否强制 Mixed）。
4. Planner（LLM）产出 `plan.json`。
5. Graph 按计划执行后续阶段并持续写 `stage_card.json`。
6. Interpreter（LLM）为每阶段生成可读结论。
7. Evaluation 完成后产出 `summary.md` + `final_summary.json`。
8. Analyst（LLM）基于 artifact 摘要回答追问。

### 3.3 Graph 节点建议

- `node_ingest`
- `node_profiling`
- `node_plan`
- `node_cleaning`
- `node_feature_engineering`
- `node_modeling`
- `node_evaluation`
- `node_final_summary`
- `node_qa`（对话模式下按需触发）

条件边：
1. `only_eda=True` 时，`node_plan -> node_evaluation`（跳过建模训练，仅输出 EDA 结论评估）。
2. `use_mixed_effect=True` 时，`node_modeling` 内部并行执行 `baseline + mixed_effect + compare`。

---

## 4. 数据契约设计

### 4.1 `column_schema.json`（输入契约）

```json
{
  "dataset_name": "optional",
  "columns": {
    "col_name": {
      "description": "含义",
      "unit": "可选",
      "suggested_role": "target|feature|group|id|time",
      "dtype_hint": "numeric|categorical|datetime|text",
      "value_range": {"min": 0, "max": 100},
      "missing_rule": "allow|forbid|special_code:-999",
      "model_hints": ["log", "skewed", "possible_outlier"]
    }
  },
  "global_hints": {
    "task_preference": "regression|classification|timeseries|survival|eda"
  }
}
```

### 4.2 `profile.json`（Profiling 输出）

关键字段：
1. `n_rows`, `n_cols`, `memory_bytes`
2. `column_types`、`missing_ratio`、`unique_ratio`
3. `time_columns`、`group_candidates`
4. `distribution_stats`（均值、偏度、峰度、异常值比例）
5. `data_structure_signals`（hierarchical/repeated/high_d_low_n 的证据）

### 4.3 `plan.json`（Planner 输出）

```json
{
  "task_type": "regression",
  "data_structure": "hierarchical",
  "target_column": "y",
  "only_eda": false,
  "use_mixed_effect": true,
  "group_column": "patient_id",
  "time_column": "visit_date",
  "feature_strategy": {
    "encoding": "onehot",
    "scaling": "standard",
    "feature_select": {"method": "model_importance", "top_k": 30}
  },
  "modeling_strategy": {
    "baseline_models": ["linear_regression", "random_forest"],
    "cross_validation": "kfold:5"
  },
  "evaluation_strategy": {
    "metrics": ["rmse", "mae", "r2"],
    "holdout_ratio": 0.2
  },
  "reasoning": [
    "schema 标记 patient_id 为 group",
    "profile 显示同一主体多次观测"
  ]
}
```

### 4.4 `stage_result.json`（阶段输出）

统一结构：

```json
{
  "stage": "cleaning",
  "run_id": "run_20260212_103045_a1b2c3d4",
  "df_key_in": "df_raw_xxx",
  "df_key_out": "df_clean_xxx",
  "actions": ["impute_median:age", "drop_duplicates"],
  "metrics": {"missing_before": 0.12, "missing_after": 0.01},
  "artifacts": [
    "storage/runs/.../cleaning/missing_report.json"
  ]
}
```

---

## 5. DataFrame 隔离与数据管理方案

### 5.1 DataManager 机制

1. 内存映射：`df_key -> DataFrame`（运行期）。
2. 可选快照：重要阶段输出 parquet 到 `storage/runs/{run_id}/snapshots/`。
3. 只允许 Tool/Skill 调用 `DataManager.get(df_key)` 取 DF。
4. `AgentState` 仅存 `df_key` 与摘要，保证序列化安全与低内存。

### 5.2 防越界策略

1. `AgentState` 使用 Pydantic `model_validator` 禁止出现 `pd.DataFrame`。
2. LLM 输入构造器统一裁剪字段白名单，防止误传原始数据。
3. 单测中加入 “state JSON 可序列化且无 DF” 的强断言。

---

## 6. LLM 设计（本地 API）

### 6.1 配置加载

`.env` 必填：

```bash
LLM_API_BASE=http://internal-llm-gateway/v1
LLM_NAME=company-model-xx
LLM_API_KEY=********
```

实现要求：
1. 缺任一变量则启动失败并给出明确报错。
2. `local_llm.py` 只依赖 `LLM_API_BASE`，不写死供应商域名。

### 6.2 三角色提示词职责

1. Planner Prompt：输入 `profile + schema + candidates + user_intent`，强制输出 JSON（可用 Pydantic parser）。
2. Interpreter Prompt：输入 `stage_result.json`，输出“关键发现/风险/下一步”卡片文本。
3. Analyst Prompt：输入 artifacts 摘要与最终总结，对用户追问给出可追溯回答（引用 artifact 路径）。

### 6.3 稳定性机制

1. JSON 解析失败自动重试最多 2 次。
2. Planner 输出经 `DecisionRules` 二次校验（如 hierarchical 必须 mixed）。
3. LLM 失败时允许回退 `mock_llm`（仅开发/测试环境）。

---

## 7. 技能编排细化

### 7.1 Profiling

输入：`df_key`  
输出：
1. `profiling/profile.json`
2. `eda/distribution_overview.png`
3. `stage_result.json` + `stage_card.json`

### 7.2 Cleaning

动作：
1. 类型修复（日期解析、类别标准化）
2. 缺失处理（按 plan 策略）
3. 异常值处理（IQR/分位截尾）
4. 去重与规则修复

输出：`df_clean_key` 与清洗报告 artifacts。

### 7.3 Feature Engineering

动作：
1. 编码、缩放、特征构造、筛选。
2. 时序任务启用 lag/rolling 特征。
3. 输出 `feature_manifest.json`（记录每个特征来源，支持解释）。

### 7.4 Modeling

#### Baseline
1. 回归：`LinearRegression / RandomForestRegressor`
2. 分类：`LogisticRegression / RandomForestClassifier`
3. 时序：轻量回归基线 + 时间特征

#### Mixed Effect（强制场景）
触发条件：`schema/profile` 判定为 hierarchical/repeated。  
执行内容：
1. 拟合 `statsmodels.MixedLM`（随机截距）
2. 导出固定效应 `fe_table.csv`
3. 导出随机效应方差 `re_variance.json`
4. 计算 `ICC.json`
5. 与 OLS 对比 `comparison_with_ols.json`

### 7.5 Evaluation

1. 按任务类型输出指标（回归 RMSE/MAE/R2；分类 AUC/F1/Accuracy）。
2. 输出误差分析图、混淆矩阵或残差图。
3. 生成 `final_summary` 与 `summary.md`。

---

## 8. 产物落盘规范

目录结构：

```text
storage/
  runs/
    {run_id}/
      profiling/
      cleaning/
      feature_engineering/
      modeling/
        baseline/
        mixed_effect/
      evaluation/
      reporting/
      logs/
```

关键文件：
1. `logs/stream.log`（可读日志）
2. `logs/events.jsonl`（结构化事件）
3. `reporting/stage_cards.json`（阶段卡片聚合）
4. `reporting/summary.md`
5. `reporting/final_summary.json`
6. `reporting/artifact_index.json`

---

## 9. Streaming Logs 与 Stage 卡片设计

### 9.1 Streaming 事件格式

```json
{
  "ts": "2026-02-12T10:30:45Z",
  "run_id": "run_20260212_103045_a1b2c3d4",
  "event": "node_start",
  "node": "modeling",
  "df_key": "df_feat_1234",
  "message": "start mixed_effect modeling"
}
```

最少事件：
1. `node_start`
2. `tool_call`
3. `artifact_saved`
4. `node_end`
5. `stage_card_emitted`
6. `error`

### 9.2 Stage 卡片模板

```json
{
  "stage": "modeling",
  "key_findings": [
    "MixedLM 显著优于 OLS（AIC 降低 12%）"
  ],
  "artifacts": [
    {"name": "ICC", "path": "storage/runs/.../ICC.json"}
  ],
  "next_step": "进入评估并生成最终报告"
}
```

---

## 10. 决策规则与 LLM 协同

### 10.1 规则优先级

1. **硬约束规则最高优先级**：如分层数据必须 Mixed。
2. Planner 在候选空间内决策，不可突破硬约束。
3. 冲突时由 `decision_rules.py` 覆盖 Planner 输出并写日志说明。

### 10.2 自动识别逻辑（建议）

1. `task_type`：
   - 连续 target -> regression
   - 二元/多类 target -> classification
   - 显式时间索引且预测未来 -> timeseries
   - 生存时间+结局事件 -> survival
   - 用户仅探索意图 -> eda
2. `data_structure`：
   - 存在 group id 且每组多观测 -> hierarchical/repeated
   - `p >> n` -> high_d_low_n
   - 其余 -> flat_iid

---

## 11. 安全与合规

1. 禁止任何公网检索调用（代码层面不引入联网搜索工具）。
2. 发送给 LLM 的 payload 做字段白名单。
3. artifact 中涉及敏感列时写脱敏摘要（可选开关）。
4. 日志默认不打印原始值，仅打印统计摘要。

---

## 12. 测试方案

### 12.1 单元测试

1. `test_data_manager.py`
   - 注册/读取/克隆/释放
   - state 中无 DF 的约束测试
2. `test_decision_rules.py`
   - hierarchical/repeated 判定
   - mixed 强制规则覆盖 Planner 冲突输出
3. `test_profiling.py`
   - profile 核心字段完整性
4. `test_mixed_effect.py`
   - 产出文件齐全（`mixed_model.pkl`、`fe_table.csv`、`ICC.json` 等）
5. `test_pipeline_smoke.py`
   - 从 ingest 到 summary 全链路执行
   - 验证 streaming logs 与 stage cards 存在

### 12.2 验收用例

1. 普通回归数据集（Flat IID）
2. 分层重复测量数据集（触发 Mixed Effect）
3. 仅 EDA 场景（跳过建模）
4. 带 `column_schema.json` 与不带 schema 对比

---

## 13. 实施里程碑（建议 8~10 天）

1. 第 1-2 天：`state/data_manager/observer/config` 骨架 + 基础测试。
2. 第 3-4 天：profiling/cleaning/feature engineering 技能闭环。
3. 第 5-6 天：baseline + mixed_effect + evaluation。
4. 第 7 天：LangGraph 编排、条件路由、错误恢复。
5. 第 8 天：Streamlit 页面、追问问答、文档与示例结果。
6. 第 9-10 天：回归测试、验收清单逐项打勾、性能和日志细化。

---

## 14. 验收对照清单（逐项打勾）

- [ ] 未使用任何公网搜索/外部工具 API  
- [ ] `AgentState` 不含 DataFrame  
- [ ] 存在并可运行 Mixed Effect 分支  
- [ ] 全流程有 Streaming Logs  
- [ ] 每个阶段都有 Stage 卡片  
- [ ] LLM 参与规划（Planner）与解释（Interpreter）  
- [ ] artifacts 全部落盘到 `storage/runs/{run_id}/`  
- [ ] 提供单测、文档、示例结果  

---

## 15. 关键实现建议（降低失败风险）

1. 先实现严格数据契约（Pydantic 模型）再接入具体算法，避免后期接口频繁返工。
2. Mixed Effect 单独先跑通最小样例，再接图编排，降低调试复杂度。
3. Planner 输出必须加 schema 验证 + 规则兜底，避免 LLM 输出漂移导致流水线中断。
4. 所有 Skill 统一返回 `stage_result.json`，保证 Interpreter 和前端展示稳定。

