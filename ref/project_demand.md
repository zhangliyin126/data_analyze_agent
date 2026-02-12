你是一个资深 Python / LangGraph 工程师。请在现有仓库中实现一个“数据分析 Agent”的 MVP。本项目采用“方案2：LLM 作为决策与解释层，LangGraph 负责稳定编排”。系统运行在公司内网环境，可以调用公司本地大模型 API，但不得调用任何公网工具或搜索服务。

====================
1. 总体架构理念
====================

系统采用四层分工：

1) Tool（原子能力）
   - 只接收 df_key 与参数
   - 不含策略判断
   - 输出 artifacts 路径与 JSON 摘要

2) Skill（阶段单元）
   - 组合多个 Tool
   - 完成一个 Stage 的确定性任务

3) LangGraph Node（编排）
   - 调度 Skill
   - 更新 State（只存 key 与摘要，不存 DataFrame）
   - 触发 Observer 事件

4) LLM（认知层）
   - Planner：从候选策略中做决策
   - Interpreter：解释阶段结果
   - Analyst：对话式问答

====================
1. 核心目标
====================

给定 CSV/Excel 数据，Agent 必须能够：

1) 自动判断任务类型  
   Regression / Classification / TimeSeries / Survival / EDA

2) 自动识别数据结构  
   Hierarchical / Repeated-measure / High-D Low-N / Flat IID

3) 结合 column_schema.json 与 profile，由 LLM Planner 生成分析计划（JSON）  
   - 选择因变量  
   - 是否使用混合效应  
   - 特征工程方向  
   - 是否仅 EDA  
   - 评估策略

4) LangGraph 按计划执行  
   Profiling → Cleaning → Feature Engineering → Modeling → Evaluation

5) 产出必须  
   - 可视  
   - 可解释  
   - 可落盘  
   - 可对话追问

6) 支持上传 column_schema.json 作为语义词典

====================
2. column_schema.json 设计与使用
====================

2.1 设计原则  
- schema 仅描述列语义与约束  
- 不强制写死目标列  
- 目标列由：  
  1) 用户提问  
  2) LLM Planner  
  在运行时确定

2.2 schema 可包含  
- 列含义、单位  
- suggested_role：target / feature / group / id / time  
- 取值范围与缺失规则  
- model_hints（如 log、skewed）

2.3 对系统的影响  
- 任务与因变量判定  
- Mixed Effect 分组识别  
- 特征工程策略  
- 卡片解释与问答

LLM Planner 必须结合 schema + profile + 用户意图做决策。

====================
3. LLM 调用方式（公司本地大模型）
====================

- 允许调用公司本地 LLM API  
- 必须通过 .env 配置：  
    LLM_API_BASE  
    LLM_NAME  
    LLM_API_KEY

- 代码中不得写死任何厂商地址  
- 禁止：  
  - 公网搜索  
  - 在线 AutoML  
  - 外部工具 API

====================
4. LLM 角色
====================

4.1 Planner  
- 输入：profile + schema + 候选策略  
- 输出：plan.json  
- 决定：因变量、Mixed Effect、特征方向

4.2 Interpreter  
- 输入：stage_result.json  
- 输出：Stage 结论卡片

4.3 Analyst  
- 基于 artifacts 回答用户提问  
- 不接触原始 DataFrame

====================
5. 技术栈
====================

- Python 3.10+  
- LangGraph / LangChain  
- Pandas / NumPy  
- Scikit-learn  
- statsmodels MixedLM（必须）  
- Matplotlib / Seaborn  
- Pydantic / PyYAML  
- 公司本地 LLM API

====================
6. 硬约束
====================

6.1 DataFrame 隔离  
- 严禁 DF 进入 AgentState  
- Tool 入参只能 df_key  
- 必须 DataManager

6.2 Artifacts  
- 必须写入 storage/runs/{run_id}/  
- 只返回路径 + JSON

6.3 上下文  
- 只传 Profile  
- 禁止全量数据入 LLM

====================
7. 混合效应模型（强制）
====================

当 schema 或 profile 表明：  
Hierarchical / Repeated-measure → 必须 Mixed Effect

至少实现：  
- Random Intercept  
- 固定效应  
- 随机效应方差  
- ICC  
- 与 OLS 对比

产出：  
storage/runs/{run_id}/modeling/mixed_effect/  
  mixed_model.pkl  
  fe_table.csv  
  re_variance.json  
  ICC.json  
  comparison_with_ols.json

====================
8. 交互体验
====================

8.1 Streaming Logs  
显示：  
- 思考步骤  
- 调用 Skill  
- df_key  
- 产出路径

8.2 Stage 卡片  
Profiling / Cleaning / Feature Engineering / Modeling / Evaluation  
含：  
- 关键发现  
- artifacts  
- 下一步

8.3 最终汇总  
- final_summary  
- 可追问  
- summary.md

====================
9. 必须实现模块
====================

src/agent/state.py  
src/agent/data_manager.py  
src/agent/graph.py  
src/agent/decision_rules.py  
src/agent/observer.py  

src/skills/profiling/dataset_profile.py  
src/skills/eda/basic_stats.py  
src/skills/eda/visualization.py  

src/skills/cleaning/*  
src/skills/feature_engineering/*  

src/skills/modeling/baseline.py  
src/skills/modeling/mixed_effect.py  
src/skills/modeling/evaluation.py  

src/skills/reporting/artifacts.py  

src/app/config_loader.py  
src/app/local_llm.py  
src/app/mock_llm.py  
src/app/agent_runner.py  
src/app/streamlit_app.py  

====================
10. 测试
====================

tests/test_data_manager.py  
tests/test_decision_rules.py  
tests/test_profiling.py  
tests/test_mixed_effect.py  
tests/test_pipeline_smoke.py  

====================
11. 验收红线
====================

以下即失败：  
- 调用公网工具或搜索  
- DF 进 State  
- 无 Mixed Effect  
- 无 Streaming Logs  
- 无 Stage 卡片  
- LLM 未参与规划与解释

====================
12. 交付
====================

需提供：  
- 可运行代码  
- 单测  
- 文档  
- 示例结果  

开始前先输出：  
- 模块清单  
- 接口签名  
- run_id 规范
