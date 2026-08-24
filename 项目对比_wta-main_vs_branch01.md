# wta-main 与 WTA-Dn-branch01 核心对比

> 生成日期：2026-08-22
> 对比对象：`/Users/fgod/Desktop/FGOD/Projects/UAS/wta-main` 与 `/Users/fgod/Desktop/FGOD/Projects/UAS/WTA-Dn-branch01`

---

## 一、两个项目分别的总结

### 1. wta-main —— 静态 WTA 精确/近似求解器（CPLEX）

- **来源**：Andersen, Pavlikov & Toffolo (2022, Annals of Operations Research) *"Weapon-target assignment problem: exact and approximate solution algorithms"* 的官方代码（`README.md`）。
- **性质**：纯数学规划项目，**没有任何强化学习/DRL 代码**（无 gym、无神经网络）。核心代码只有两个文件：`src/wta_cplex.py`（985 行，全部算法）和 `src/validator.py`（独立解验证器）。
- **问题模型**：静态单阶段 WTA。m 个武器、n 个目标、每武器 μ 发弹；毁伤概率 p[i,j]；目标函数为**非线性**的期望存活目标价值：
  min Σⱼ wⱼ · Πᵢ (1−p[i,j])^x(i,j)
  通过分段线性下近似（误差界 δ，断点由 broyden1 求根，`wta_cplex.py:251`）线性化后交给 CPLEX。
- **三种算法**（`-approach` 参数）：
  | 算法 | 类 | 位置 |
  |---|---|---|
  | branch-and-adjust（默认） | `WTA_BranchAdjust_Int` | `wta_cplex.py:676` |
  | probchain（概率链） | `WTA_ProbChain` | `wta_cplex.py:316` |
  | underapprox | `WTA_UnderApprox` | `wta_cplex.py:527` |
- **数据格式**：`data/wta_{m}x{n}x{μ}.txt`，首行 `m n mu`，n 行目标权重，m×n 行 `i j p[i,j]`。31 个算例，规模 10×20 到 500×1000。
- **流程**：CLI 求解 → 打印 gap/上下界 → `validator.py` 独立验证。已完成的 `wta_150x300x3` 实验显示 600s 内 branch-and-adjust gap 44.09%、probchain gap 93.17%（均超时未证明最优）。

### 2. WTA-Dn-branch01 —— 动态 WTA（DWTA）仿真 + LLM 辅助求解分支

- **性质**：在 main 的静态求解器之上，叠加了"**多波次动态目标到达 + 距离逼近 + 突防泄漏**"的仿真层，以及 **LLM（DeepSeek）辅助求解策略层**。同样不是 DRL 项目。
- **核心原则**：**原求解器从不被修改**，逐波以 subprocess 调用 `wta_cplex.py`（`src/dwta/wave_runner.py` 头注释；唯一小改动是新增 `-warmstart` 参数，`wta_cplex.py:722`）。
- **动态场景复杂化点**（核心实例 `data/dyn_wta_50x100x3_K10_dist.txt`，50 武器 × 100 目标 × K=10 波 × μ=3）：
  1. **多波次到达**：100 目标均衡分 10 波，每波 10 个新目标（`gen_dynamic_data.py`）；
  2. **距离逼近**：目标初始距离 d0∈{2..10} km，每留任一波 −1 km（`instance.py:136-144`）；
  3. **时空耦合命中率**：p_eff(i,j,k) = min(pcap, p_ij · d0_j / d_j(k))，越近越好打（`instance.py:146-155`）——形成"早打效率低 vs 晚打突防风险"的权衡；
  4. **停留上限 L=3 与突防**：活过 L 波的目标按全权重泄漏离场（`simulator.py:148-159`），泄漏率是首要指标。
- **求解方式**：滚动时域（rolling horizon）——每波组装当前目标集的临时静态子实例 → CPLEX 求解 → 伯努利杀伤结算 → 下一波。
- **LLM 辅助层**（`src/dwta/llm_agent.py`，DeepSeek API，两段式）：
  - 波前模块 a：建议分配作为 CPLEX MIP **warm start**；模块 b：求解器参数自适应（白名单）；模块 d：延迟打击低价值目标；
  - 波后模块 c：中文战况点评；
  - 严格校验护栏（`llm_agent.py:157-226`），逐字段降级，最坏回退纯 CPLEX 基线。
- **等价的状态/动作/奖励结构**（虽非 RL，但接口已就绪）：state（`simulator.py:116-130`）、`decide(state) -> 分配方案`（`simulator.py:24`，为 RL/LLM 策略预留的纯函数注入点）、终局指标为总泄漏价值/泄漏率。
- **入口**：`src/run_demo.py`（`--policy base|llm`、`--smoke`）；数据生成 `src/gen_dynamic_data.py`（`--input 静态 --waves 10 --dist --L 3 --pcap 95`）。

---

## 二、核心区别（直接回答你的关键问题）

### ★ branch01 是否在数据集复杂化的前提下，实现了 main 已实现的算法？

**是的，且实现方式对你非常有利。**

1. **原算法原样保留**：branch01 中的 `src/wta_cplex.py` 就是 main 的那份求解器（branch-and-adjust / probchain / underapprox 三种 approach 全部可用），唯一新增是 `-warmstart` 参数。动态仿真层通过 subprocess 逐波调用它，**没有重写、没有改动任何算法逻辑**。
2. **复杂的是"数据/场景"，不是"算法"**：branch01 的复杂化全部发生在实例层（波次、距离、停留上限、突防），每波求解时会被 `wave_runner` 压缩回一个 main 格式的静态子实例喂给原求解器。也就是说：**动态数据集 ⊃ 静态数据集，原算法在动态框架内是逐波复用的**。
3. **数据生成链路已经存在**：`gen_dynamic_data.py` 就是"main 静态算例 → 拓展动态算例"的现成转换器（加波次划分、距离扩展、L、pcap），且向后兼容静态格式。

### 对你的计划（main 数据集 → 适当拓展 → 复现原算法 + 创新算法）的评估

**可行性高，branch01 已经把你要的前 80% 基础设施搭好了：**

- "用 main 的数据集进行适当拓展" → `gen_dynamic_data.py --input <main静态算例> --waves K --dist --L --pcap` 一条命令完成，main 的 31 个算例（10×20 ~ 500×1000）都可转换；
- "在拓展数据集上复现原算法" → branch01 的滚动时域 + subprocess 框架已经做到，跑 `run_demo.py --policy base` 即是原算法在动态数据上的复现；
- "创新算法的实现" → `simulator.decide(state)` 是预留的**纯函数策略注入点**，签名即状态进、分配出——无论是接入新的启发式、DRL（DQN/PPO）还是改进的 LLM 策略，都只需实现这个函数，不用动仿真器和求解器。

**需要注意的风险/边界**：

- **CPLEX 规模限制**：社区版 1000 变量上限，大算例（如 50×100 动态实例）会触发 Error 1016 直接退出（`wave_runner.py:24-32`）——拓展 main 大算例前先确认许可证；
- **决策粒度变了**：main 中算法是一次性全局最优（对静态实例）；branch01 中它退化为"每波局部最优"的贪婪式滚动策略，**没有跨波前瞻**——这正是创新空间所在（如跨波前瞻、值函数近似、学习型策略）；
- **弹药语义**：branch01 目前是"每波每武器重置 μ=3"（需求文档决策 D2），与 main 的"总弹药预算"语义不同；数据格式首行已预留扩展字段支持全局库存，但代码未实现；
- **两个项目都没有 RL**：如果你的"创新算法"指 DRL，需要自己从零实现（好在 decide 接口和 state 结构就是按 MDP 设计的）。

### 其它重要区别一览

| 维度 | wta-main | WTA-Dn-branch01 |
|---|---|---|
| 问题类型 | 静态单阶段 WTA | 多波次动态 WTA（DWTA） |
| 数据格式首行 | `m n mu` | `m n K mu L pcap`（向后兼容） |
| 目标函数 | 期望存活价值（非线性，分段线性化） | 终局泄漏价值/泄漏率 + 每波期望存活价值 |
| 随机性 | 无（确定性优化） | 伯努利杀伤结算，Monte-Carlo 多种子评估 |
| 求解架构 | 一次性 CPLEX 求解 | 滚动时域：逐波临时实例 + subprocess CPLEX |
| 智能辅助 | 无 | LLM（DeepSeek）warm start / 参数自适应 / 延迟打击 / 战况点评 |
| 关键机制 | 分支回调、目标修正回调 | 距离-时间耦合命中率 p_eff、停留上限 L、突防泄漏 |
| 评估方式 | gap / 上下界 / validator 验证 | 多 seed 泄漏率、突防数、建议吸收率、token 消耗 |
| 策略扩展接口 | 无 | `simulator.decide(state)` 纯函数注入点（预留 RL） |
| 核心规模实验 | 150×300×3（600s，gap 44%） | 50×100×3、K=10 波（每波 60s） |

---

## 三、关键文件索引

**wta-main**
- `src/wta_cplex.py` — 全部算法（BranchAdjust :676 / ProbChain :316 / UnderApprox :527）
- `src/validator.py` — 独立解验证
- `HANDOFF.md`、`实验报告_wta_150x300x3.md` — 实验操作与结果

**WTA-Dn-branch01**
- `src/dwta/instance.py` — 动态实例解析（含 p_eff、距离推进）
- `src/dwta/wave_runner.py` — 逐波 subprocess 调用原求解器
- `src/dwta/simulator.py` — 波次循环、伯努利结算、`decide()` 注入点
- `src/dwta/llm_agent.py` — LLM 两段式策略层
- `src/gen_dynamic_data.py` — 静态 → 动态数据转换 CLI
- `src/run_demo.py` — 仿真主入口
- `README-DYNAMIC.md`、`需求文档_动态WTA扩充与大模型辅助求解.md`、`实验报告_组会汇报版.md` — 设计与实验文档
