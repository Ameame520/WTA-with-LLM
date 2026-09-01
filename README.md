# WTA-Dn-branch02 — DN-WTA 多智能体动态武器-目标分配（MARL 主实验）

DN-WTA 多智能体时序环境 + 基线策略 + 数据集与评估协议。本分支为 **MARL 主实验**的干净基础：主实验与主框架的结果将作为论文/专利发表的基础。

## 目录结构

```
WTA-Dn-branch02/
├── main.py                  # 全流程运行入口（唯一入口，DN-WTA 管线）
├── README.md                # 本文件
├── 项目整理规范.md           # 项目结构规范，重构前必读
├── cplex/                   # 静态 WTA 精确求解器（仅作集中式上界基线与最优性间隙参考；MARL 不依赖）
├── dwta/                    # 核心框架（数据层 / 多智能体时序环境 / 基线策略 / 求解器子进程机制）
├── experiments/             # 实验支撑（数据生成 / split 批量评估 / 指标报告）
├── data/                    # 算例数据（dn-data-v3 主基准，见 MANIFEST）
├── logs/                    # 正式实验终端日志
└── output/                  # 实验结果，每次实验一个子文件夹
```

一个顶层文件夹 = 一个模块；目录细则见 `项目整理规范.md`。

## 快速开始

所有命令**在项目根目录**执行。`none` / `greedy` 策略只需 numpy；`cplex` 策略与 `--dn-reference` 需要安装 CPLEX 的解释器（默认 `/opt/anaconda3/envs/wta/bin/python`，可用 `--python` 覆盖）。

```bash
# 冒烟测试（小算例，验证链路，不需要 CPLEX）
python main.py --instance data/dy-data-v1/dn_3x10_K10_s1.txt --policy greedy --seeds 3

# 无防御基线（复现数据集说明 §5 的泄漏直方图）
python main.py --instance data/dn-data-v3/dn_3x50_K10_s01.txt --policy none --seeds 30

# 分布式局部贪心（仅观测接口 §7，与 MARL 策略同一信息边界）
python main.py --instance data/dn-data-v3/dn_3x50_K10_s01.txt \
    --policy greedy --seeds 30 --dn-reference

# 集中式短视 CPLEX 上界基准（gap 恒为 0，可作自检）
python main.py --instance data/dn-data-v3/dn_3x50_K10_s01.txt \
    --policy cplex --seeds 30 --timelimit 30

# v3 实例族 split 批量评估（读 MANIFEST 固定划分；正式对比协议：test × 30 seeds）
python experiments/dn_family_eval.py --split test --policy cplex \
    --seeds 30 --seed-base 42 --timelimit 30 --output output/e13_dn3_cplex

# 生成 DN-WTA 数据实例（缺省即 v3 规范参数）
python experiments/gen_dn_data.py --seeds 1,2 --outdir data/tmp_check
```

每次运行输出 `report.json` / `report.md`（族评估输出 `family_report.*`），并自动清理临时文件。

## MARL 接入点

- **环境**：`dwta/dn_env.py` — `DNEnv` 提供逐步交互：`get_observation(i, t)` 局部观测（§7 信息边界）、`can_fire / fire` 动作执行、延迟毁伤结算与全局弹药池；`simulate_dn(dn, seed, policy, log)` 为整回合蒙特卡洛循环。
- **策略接口**：`dwta/dn_policies.py` — 任何策略实现 `policy.act(env, t) -> (actions, info)`（`actions: {agent_i: target_j 或 None}`）。`GreedyPolicy` 即该接口的分布式局部求解参考实现（仅用观测、无通信、无求解器）。
- **评估协议**：`data/dn-data-v3/MANIFEST.md` — train s03–s26 / val s27–s30 / test s01–s02 固定划分；MARL 在 train 训练、val 调参、test（2 实例 × 30 seeds）终评，并报告泛化代价（指标⑩）。

## 模块说明

| 模块 | 内容 |
|---|---|
| `cplex/` | 静态求解器（分支定界 + 分段线性，**算法零改动**），`validator.py` 校验解的可行性；仅被基线策略经子进程调用 |
| `dwta/` | `dn_instance.py` DN 算例解析与动力学推导（纯数据层）；`dn_env.py` 多智能体时序环境（信息边界 §7）；`dn_policies.py` none / greedy（局部求解）/ cplex（上界）三基线；`wave_runner.py` 求解器子进程机制 |
| `experiments/` | `gen_dn_data.py` 数据生成器（缺省 v3 参数，`--w-trend 0 --mu 8 --quota 2,2,1` 可复现 v2）；`dn_family_eval.py` split 批量评估；`dn_report.py` 指标聚合与报告 |

## 文档索引

- `项目整理规范.md` — 目录结构与重构红线
- `DN-WTA_v3_数据集说明.md` — 当前基准数据集规范
- `DN-WTA_v2_数据集说明.md` — v2 遗留规范（环境动力学 §4/§7 的历史定义）
- `评价指标体系_精简三层版.md` — 指标规范（`experiments/dn_report.py` 的实现依据）
- `data/dn-data-v3/MANIFEST.md` — 固定 split 划分与评估协议
