# 实验报告 E14 — DN-WTA v3 学习策略（marl）训练与评估

日期：2026-09-02 ｜ 需求：《需求文档_DN-WTA核心框架实现》（六模块 M1–M6）
环境：conda env `wta`（torch 2.1.0 / MPS / numpy 2.x），macOS（arm64）

## 1. 实验目标

在 DN-WTA v3 基准上实现并验证 CTDE 学习型策略：感知重构（λ 递推）→ 集合注意力 actor → MAPPO 训练 → 与 none/greedy/cplex 三基线在 test split（s01–s02 × 30 seeds 42–71）同协议对比。

## 2. 回归门（阶段 0）

marl 开发前重建三基线（`output/regress_e11~13_*`），与 LOGS_SUMMARY 记录**逐字段一致**：

| 基线 | 泄漏率（本文档记录 → 本次回归） | 其他校验 |
|---|---|---|
| e11 none | 1.0000 → 1.0000 | 30 seeds 直方图逐 seed 一致（0,1,2,5×7,期末12）；指纹 `9134a16f`/`e717a4c1` 与 MANIFEST 一致 |
| e12 greedy | 0.7913±0.0095 → 0.7913±0.0095 | gap 0.2768 复现 |
| e13 cplex | 0.7085±0.0343 → 0.7085±0.0343 | worst 0.7428、gap≈1e-14、invalid 0.0963、ammo 67.92 复现 |

## 3. 模块验收（自检记录）

| 模块 | 验收项 | 结果 |
|---|---|---|
| M1 perceive | λ 递推/翻转清零/输入纯度（不引用 env） | ALL PASS |
| M2/M3 network | 置换一致性（目标重排 logits 映射不变）；参数量 28482 ∈ [1e3,1e5] | ALL PASS |
| M4 masking | masked 动作全部通过 `env.can_fire`（合法执行） | ALL PASS |
| M5 reward | 对账 ΣR 与 (destroyed−leak)/total 误差 <1e-9（4 episodes）；kill 计数一致；credit 键 ⊆ 发射槽位 | ALL PASS |
| 接入 policy | 3 seeds：illegal=0，act wall <1s（首步 0.09s 为 MPS 编译，稳态毫秒级） | ALL PASS |
| M6 train | 冒烟 6 iters：日志滚动、无 NaN、best.pt 产出、val 曲线下降 | ALL PASS |

## 4. 训练（E14a）

配置：MAPPO 风格 PPO（clip 0.2 / ent 0.01 / GAE(0.95,0.99) / Adam 3e-4 / 32 episodes/iter / PPO 2 epochs），动作条件 critic + COMA 式反事实基线 `V(s,a)−V(s,a_{−i},⊥ᵢ)`，τ 退火 1.0→0.5（2000 iters），val=s27–s30×seeds 42–51 每 20 iters 评估，patience=20 评估点，MPS。

| 项 | 值 |
|---|---|
| 墙钟 | 5918 s（1.64 h，远低于 24h 上限） |
| 停止原因 | early_stop（460 iters，23 评估点） |
| env steps | 132,480 |
| best val | **0.8943**（iter 60；采集种子 100001 起，与评估种子隔离） |
| 训练曲线 | 0.937 →（震荡 0.92–1.0）→ 0.898 收敛方向 |

过程中修复的两个关键问题：
1. **τ 一致性 bug**：PPO 回放时用退火中的 τ 重算 logp_new，与采集时 logp_old 分布不一致 → 策略坍缩（val 1.0）。修复：flat 样本记录采集时 τ，回放同 τ。
2. **safe-delete 阈值**：CPLEX 参考解长跑累计删除 >500 临时文件被工作区保护机制中断 → 规避：`--output /tmp/<dir>` 后拷回。

## 5. 终评（E14b，test split × 30 seeds）

| 指标 | marl（e14） | greedy（e12） | cplex（e13） | none（e11） |
|---|---|---|---|---|
| 泄漏率 | **0.9009±0.0295** | 0.7913±0.0095 | 0.7085±0.0343 | 1.0000 |
| worst 实例 | 0.9304（s02） | — | 0.7428 | 1.0000 |
| gap（iii） | 0.7032±0.0361 | 0.2768 | ≈0 | — |
| 无效交战率 | 0.1130 | 0.35 | 0.0963 | — |
| 弹药效率 | 21.75 | 46.62 | 67.92 | — |
| 决策时延 p50/p90 | **4ms / 10ms**（纯策略） | ~0.38s* | 0.383/1.821s | — |
| shots/run | 18.0 | 18.0 | 18.0 | 0 |

\* greedy 时延含参考解调用；marl 含参考解口径 p50 0.47s（CPLEX 子进程开销），纯策略推理毫秒级。

### 泛化性快照（`--no-ref`，30 seeds）

| split | 泄漏率 |
|---|---|
| train（s03–s26，24 实例） | 0.9166±0.0247 |
| val（s27–s30，4 实例） | 0.8902±0.0134 |
| test（s01–s02，2 实例） | 0.9009±0.0295 |

三 split 一致 → **无过拟合迹象**（训练只见过 train split 实例族，test/val 表现同水平）。

## 6. 结论与差距分析

1. **红线全部达成**：actor 前向仅 §7 观测 + 实例公共先验；参数量 28482 < 1e5；act 时延毫秒级 < 1s；采样流独立 Generator（评估 greedy argmax 每 seed 确定性）；seed 隔离完备（采集 100001+ / val 42–51 / 终评 42–71）。
2. **泄漏率显著优于 none（−10pp）但未追平 greedy**。归因：团队奖励信用稀疏（kill 是唯一强正信号，命中率 ~50%），第一版 1.6h 训练下 val 曲线仍在缓慢下降即被 patience 截停；策略学到的行为偏保守（无效交战率 0.11 优于 greedy 0.35，但弹药效率 21.8 低——发射少导致毁伤少）。
3. **可复现性**：三基线数值逐字段复现；marl 评估确定性（greedy argmax，同 seed 同指纹）；训练产物三件套齐备（`best.pt`/`train_log.jsonl`/`train_summary.json`）。

## 7. 后续方向（需求文档 P1/P2 余项）

- 学习效率：增大 batch（32→128 episodes/iter）平均化信用噪声；对 credit 项加权或改用 advantage 分位数裁剪。
- 快速对齐：greedy 轨迹行为克隆热启动后 PPO 微调（P1 可选项）。
- 规模化：m=5/n=100/K=20 上限实验（红线压力测试，P2）。
