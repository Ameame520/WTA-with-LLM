# DN-WTA v1 下一步工作规划

> 制定日期：2026-08-25 ｜ 前置：DN 数据集已生成并验证（见《DN-WTA_v1_数据集说明.md》）
> 依据：`数据集规则.md` §11–§14，结合现有实现（`dwta/`、`cplex/`、`llm/`、`experiments/`）落地
> 总标准：**复用已有项目实现 + 全部预期评价指标可实现 + 提供统一框架进行多算法公平对比**

---

## 0. 目标与验收标准

| # | 目标 | 验收标准 |
|---|---|---|
| G1 | 每个武器平台 = 独立 Agent，可跑 MARL | IPPO / MAPPO 在统一环境上训练收敛并产出评估结果 |
| G2 | 全局信息 + 局部观测 + 每步末信息同步 | 环境实现 §11 八步执行序与 §12.3 观测边界，且有防泄密单测 |
| G3 | 全局性系统约束（弹药全局扣减、摧毁全局生效、突防全局移除） | 对应单测全部通过 |
| G4 | 多算法统一环境公平对比（MARL / MARL+LLM / 求解器 / 贪婪 / 启发式） | 所有算法仅通过 `get_observation(i,t)` 获取信息；表 1/表 2 指标在同一 runner 下自动产出 |

**信息公平性红线**（§13）：分布式算法（MARL/贪婪/启发式）统一走局部观测；CPLEX 为"集中式当前状态上界"，单独分组报告，不与分布式算法做同信息条件对比。

## 1. 总体架构

```
                    ┌────────────────────────────────────────────────┐
                    │  DNEnv（dwta/dn_env.py，新增，唯一仿真环境）      │
                    │  · Global State（§12.2）: 弹药/在途弹/摧毁/历史  │
                    │  · get_observation(i,t)（§12.3）信息边界        │
                    │  · step(actions) 内部按 §11 八步序执行          │
                    │  · 结果结构 v2 落盘（指标自动可算）              │
                    └──────────────┬─────────────────────────────────┘
                                   │ 统一 Policy 接口: act(obs_i) → a_i
        ┌──────────────┬───────────┼──────────────┬───────────────────┐
        ▼              ▼           ▼              ▼                   ▼
  CPLEX 集中式上界  贪婪(分布式)  启发式(分布式)  IPPO/MAPPO        MARL+LLM
  (复用 wave_runner (dwta/       (dwta/          (experiments/      (llm/dn_advisor.py
   子进程零改动)    dn_policies)  dn_policies)    dn_train.py)       复用审计/降级)
```

- 所有算法消费**同一个** `DNEnv`，同一 RNG 流协议 → 同 seed 逐位可复现，QA 门沿用；
- 奖励、结算、指标统计全部在环境层实现一次，算法层只做决策——避免旧架构中"指标散落在各实验脚本"的问题。

## 2. 阶段 A：统一多智能体环境（M1，最高优先级）

### 2.1 位置与依赖

- 新文件 `dwta/dn_env.py`（属模块②"仿真核心"，符合 `项目整理规范.md`）；
- 依赖仅 `dwta/dn_instance.py` + numpy（训练侧另需 torch，见 D8）；
- 旧 `simulator.py`/`wave_runner.py`/`main.py` 旧链路**不动**，保证 E1–E4 可复现。

### 2.2 API 规格

```python
env = DNEnv(dn_instance, seed=42)
obs, share_obs = env.reset()            # obs: {i: Obs_i}；share_obs 供 CTDE critic
actions = {i: policy_i.act(obs[i])}     # 每步每 Agent 最多发射 1 枚或保持
obs, share_obs, rewards, done, info = env.step(actions)
```

**动作空间**：`Discrete(n+1)`——`0` = 不发射；`j+1` = 对目标 `j` 发射。
**动作掩码**（合法 = 已出现 ∧ 存活 ∧ 未突防 ∧ B_i>0）。注意：**允许**对"已有在途弹覆盖"的目标再射——禁止即泄密（Agent 本就不该知道别家的在途弹，D5）。

**执行序**（§11 严格落地）：目标出现 → 生成观测 → 各 Agent 并行决策 → 执行发射（锁 `p_shot`、扣弹药）→ 推进 Δt → 结算到达的拦截弹（Bernoulli，独立）→ 突防判定（同时刻先拦截后突防）→ 信息同步（摧毁/突防状态全局广播）。

**尾部结算（D1，默认：结算）**：t ≥ K 后无决策步，但已发射的在途弹继续结算至 `max t_hit`；期间不产生新突防（口径已确认：期末存活不计泄漏）。

### 2.3 观测编码（D4）

每个 Agent 的观测为定长特征向量（面向神经网络）+ 结构化 dict（面向 LLM/规则算法）：

- 共享块：t（归一化）、K；已出现目标的 `[w_j, r_j(t), 存活0/1, 已出现步数]`（按目标槽位，未出现置零掩码）；
- 自身块：`B_i/mu`、对已出现目标的 `[p_ij, d_ij(t), p_eff_ij(t), h_ij(t)（预计飞行步数）, ETA_j=自身在途弹数]`；
- 掩码向量：当前合法动作集。

**CTDE**（§14）：训练阶段 critic 输入 `share_obs`（当前时刻已出现目标的联合状态 + 各 Agent 弹药）；执行阶段 actor 仅用 `obs[i]`；训练/执行均不得触碰未出现目标信息。

### 2.4 奖励设计（D3，默认方案 + 消融）

- 默认：**全局团队奖励** `r_t = Σ(当步确认摧毁的目标价值) − Σ(当步突防目标价值)`（稀疏事件奖励，全局一致，符合 G3"全局性系统"）；
- 消融项：发射成本系数 `−λ·发射数`（λ∈{0, 0.01·平均w}），检验是否催生更省弹策略（联动指标④）。

### 2.5 确定性契约

RNG 消费顺序固定：结算按 `(t_hit, 发射序号)` 排序，同目标多弹按序 Bernoulli；同参数两次运行 `result_hash` 一致（QA 门，沿用旧协议）。

### 2.6 环境验收单测（E5 的核心）

1. 弹药全局扣减：发射后 B_i 跨步递减、永不恢复；弹药为 0 时动作被掩；
2. 摧毁全局生效：目标被结算摧毁后，所有 Agent 下一同步步观测一致；
3. 延迟结算：`p_shot` 锁定值 ≠ 结算时刻 `p_eff`（远距发射近距结算场景断言）；
4. 无效交战：目标已毁后在途弹失效且弹药不返还（⑰ 可统计）；
5. **防泄密**：构造观测序列断言——未出现目标槽位全零、观测中不包含其他 Agent 私有量；
6. 可复现性：同 seed 双跑逐位一致。

## 3. 阶段 B：算法适配层（统一 Policy 接口）

所有算法实现 `act(obs_i) → action`（+ 可选 `observe(reward, done)`），由同一 runner 驱动。

### B1 集中式 CPLEX 上界（复用，零改动）

- 每时间步：环境将**当前已出现目标的联合状态**（当前 `p_eff`、各平台剩余弹药）写成静态子算例（每武器单发上限 → mu_i=1），经 `wave_runner.run_solver` 子进程调用 `cplex/wta_cplex.py`——旧求解器与调用链完全复用；
- 角色定位（§13）：Centralized Current-State Upper Bound，用于 ③ 最优间隙参考 + 集中式对照列，**不进入分布式公平组**；
- 已知偏差（写入报告注脚）：它看不到别家在途弹的未来结算，会对已注定被毁的目标重复分配 → 上界偏松；可选增强（D2b）：把在途弹期望生存率折算进子算例目标值。

### B2 分布式基线（`dwta/dn_policies.py`，纯局部信息）

| 策略 | 规则 |
|---|---|
| Random-legal | 合法动作均匀随机（冒烟 + 下界） |
| Greedy | `argmax_j w_j·p_eff_ij(t)`，弹药>0 且有合法目标即射 |
| Heuristic | 威胁紧迫度优先：`score_j = w_j·p_eff_ij(t) / max(1, 剩余突防步数)`，超阈值才射；弹药保留规则（末段省弹） |

### B3 MARL（`experiments/dn_train.py`）

- IPPO（独立学习，参数不共享）与 MAPPO（CTDE，centralized critic 吃 `share_obs`）双实现；
- 训练实例默认 `dn_5x20_K10_s1` / `dn_10x50_K10_s1`（中小规模），大实例只做零样本评估（⑩）；
- 训练日志产出 ⑨（收敛曲线 vs ③）⑪（墙钟/参数量/显存）。

### B4 MARL+LLM（`llm/dn_advisor.py`，复用 llm/ 的审计与降级机制）

- 角色映射（旧模块 → 新语义）：a 热启动 → **动作先验**（LLM 目标优先级 → logit 偏置）；d 延迟建议 → **保持火力建议**（关键步触发）；c 解释 → 每步/关键事件中文解释（定性附录）；
- 全部调用落 `llm_calls.jsonl`（⑫⑬⑮ 自动可算）；失败自动降级回纯 MARL 路径，仿真不中断；
- ⑭ 采纳率 = LLM 建议与最终动作一致率（分模块统计）。

## 4. 阶段 C：指标与报告（`experiments/dn_report.py` 或扩展 `report.py`）

- 从**结果结构 v2** 自动计算表 1（①–⑧）+ 表 2（⑨–⑯）+ 可选（⑰⑱）；
- 结果结构 v2（多智能体版最小契约，扩展原 §6）：

```json
{
  "meta": {"instance": "dn_5x20_K10_s1.txt", "m": 5, "n": 20, "K": 10, "mu": 2,
           "dt": 2.0, "delta_d": 1.0, "v_m": 1.5, "pcap": 0.95, "total_value": 1186,
           "policy": "mappo", "seed": 42},
  "steps": [{
    "t": 0,
    "actions": {"0": 3, "1": 0, "2": 7},
    "valid": {"0": true, "1": true, "2": true},
    "wall_time": 0.012,
    "fired":  [{"i": 0, "j": 2, "p_shot": 0.44, "t_hit": 3}],
    "settled": [{"i": 0, "j": 2, "killed": true}],
    "leaked":  [],
    "expected_cost": 41.2,
    "llm": {"calls": 1, "tokens_prompt": 2100, "tokens_completion": 180,
            "duration_s": 3.4, "accepted": true, "degraded": false}
  }],
  "summary": {"leak_value": null, "breakthrough_count": null, "shots_total": null,
              "coverage_threatened": null, "mean_engagement_age": null,
              "void_rate": null, "final_ammo": null}
}
```

- `p_shot` 逐发射记录 → ② 解析重算、⑰ 无效交战率均可离线复算；
- Pareto 图（⑦×②）为主图；汇总表沿用表 1/表 2 版式。

## 5. 阶段 D：实验矩阵

| 实验 | 内容 | 实例 | 种子 |
|---|---|---|---|
| E5 | 环境验收（§2.6 单测 + Random-legal/Greedy 冒烟） | dn_3x10_K10_s1 | 42–44 |
| E6 | 分布式基线 + 集中式上界全量评估 | {3x10, 5x20, 10x50, 20x100}×{s1,s2} | 42–44 |
| E7 | MARL 训练 + 网格零样本评估（⑩） | 训练 5x20/10x50；评估全网格 | 42–44 |
| E8 | MARL+LLM 消融（先验/延迟/解释） | 同 E7 训练实例 | 42–44（配对） |
| 汇总 | 表 1/表 2 + Pareto + 消融终表 | — | — |

## 6. 目录与入口规划（遵守《项目整理规范.md》）

```
新增文件（全部平铺，无新顶层文件夹）：
dwta/dn_env.py          # 环境核心（模块②扩展）
dwta/dn_policies.py     # 分布式基线策略
experiments/dn_train.py # MARL 训练入口
experiments/dn_report.py# v2 指标/报告（或并入 report.py）
llm/dn_advisor.py       # LLM 顾问（复用 llm_agent 审计设施）
main.py                 # 扩展 --dataset dn 子命令（保持唯一入口，D7）
```

依赖方向保持单向：`main → dwta/llm/experiments`；`dn_env → dn_instance`；`dn_train → dn_env`；旧链路零改动。

## 7. 待确认决策点

| # | 决策 | 默认提案 |
|---|---|---|
| D1 | 尾部结算（t≥K 后在途弹） | 继续结算至 max t_hit，无决策步、不产生新突防 |
| D2 | CPLEX 上界每步时限 | 10 s/步（与旧 60 s 区分，控制 E6 墙钟）；D2b：先不折算在途弹，报告标注"近视上界" |
| D3 | 奖励 | 全局团队奖励（摧毁−突防），λ 发射成本消融 |
| D4 | 观测编码 | 定长向量（NN）+ dict（规则/LLM）双输出，槽位掩码 |
| D5 | 动作掩码 | 已出现∧存活∧未突防∧B_i>0；允许对在途弹覆盖目标再射 |
| D6 | 上界运行范围 | 全部网格实例；20x100 若触社区版变量上限则裁剪目标集并标注 |
| D7 | 入口 | 扩展 main.py（`--dataset dn --policy {cplex,greedy,heuristic,random,mappo,ippo,llm}`） |
| D8 | RL 依赖 | PyTorch + 自研轻量 IPPO/MAPPO（不引 PettingZoo/EPyMARL，保持依赖最小、可控可复现） |

## 8. 里程碑与风险

| 里程碑 | 内容 | 预估 |
|---|---|---|
| M1 | dn_env + 单测 + E5 | 1 周 |
| M2 | 基线/上界 + E6 | 3–5 天 |
| M3 | MARL 训练 + E7 | 1–2 周 |
| M4 | LLM 融合 + E8 | 1 周 |
| M5 | 汇总报告（表 1/表 2/Pareto） | 2–3 天 |

| 风险 | 影响 | 缓解 |
|---|---|---|
| CPLEX 社区版 1000 变量限制（20x100 每步子算例可达 2000 变量） | 上界缺失 | 裁剪子算例目标集（top 威胁）；或该规格仅报分布式组 |
| LLM 时延远超 dt=2s 实时阈值 | ⑦ 口径争议 | ⑦ 分列"决策时延"与"LLM 增强时延"，Pareto 图分层标注；LLM 定位为"训练/离线增强"而非在线 |
| MARL 在稀疏奖励下不收敛 | E7 延期 | 事件奖励已较密（每步可能有结算）；备选 reward shaping（势函数 = 剩余威胁期望） |
| 信息边界实现 bug（隐性泄密） | 公平性失效 | §2.6-5 防泄密单测作为 CI 红线；观测构造集中在 env 单点 |
| 网格全量训练算力不足 | E7/E8 墙钟爆炸 | 只训中小实例，大实例零样本评估（⑩ 本就要求如此） |
