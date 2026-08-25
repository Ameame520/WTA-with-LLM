# WTA-Dn — 动态武器-目标分配（DWTA）

静态 WTA 的分支定界精确求解器 + 动态多波次蒙特卡洛仿真 + LLM 辅助决策的实验框架。

## 目录结构

```
WTA-Dn-branch01/
├── main.py            # 全流程运行入口（唯一入口）
├── README.md          # 本文件
├── 项目整理规范.md     # 项目结构规范，重构前必读
├── 实验综合报告.md     # E1–E4 实验结论
├── cplex/             # 模块①静态 WTA 精确求解器（wta_cplex.py + validator.py，核心算法）
├── dwta/              # 模块②动态多波次仿真核心（instance / simulator / wave_runner）
├── llm/               # 模块③LLM 辅助决策（llm_agent.py）
├── experiments/       # 模块④实验支撑（gen_dynamic_data 数据生成 / report 报告生成）
├── data/              # 算例：动态 dyn_*.txt、DN-WTA v1 网格 dn_*.txt（32 个）
├── logs/              # 正式实验终端日志（索引见 logs/LOGS_SUMMARY.md）
└── output/            # 实验结果，每次实验一个子文件夹（索引见 output/README.md）
```

一个顶层文件夹 = 一个模块；文件夹内不再嵌套子文件夹（output 的实验文件夹、data 除外），详见 `项目整理规范.md`。

## 快速开始

所有命令**在项目根目录**执行；Python 环境需安装 CPLEX（默认解释器 `/opt/anaconda3/envs/wta/bin/python`）。

```bash
# （旧流程）生成动态算例：静态算例 → K 波次动态算例；旧静态 wta_*.txt 已于 2026-08-25 清理，需自备
# python experiments/gen_dynamic_data.py --input <your_static.txt> --waves 10 --seed 7

# 生成 DN-WTA v1 数据集（MARL 多智能体对比网格：m{3,5,10,20} × n{10,20,50,100} × 2 seeds）
python experiments/gen_dn_data.py --grid

# 冒烟测试（小算例快速验证链路，不依赖 LLM API）
python main.py --smoke --seeds 3 --timelimit 30

# 正式实验
python main.py --instance data/dyn_wta_50x100x3_K10_dist.txt \
    --seeds 3 --seed-base 42 --timelimit 60 --policy base          # E1 纯 CPLEX 基线
python main.py --instance data/dyn_wta_50x100x3_K10_dist.txt \
    --seeds 3 --seed-base 42 --timelimit 60 \
    --policy llm --llm-modules a+b+c+d --llm-timeout 300           # LLM 全模块
```

`main.py` 每次运行输出 `report.json` / `report.md`（LLM 实验另有 `llm_calls.jsonl` 审计），并自动清理波次临时文件。

## 模块说明

| 模块 | 内容 |
|---|---|
| `cplex/` | 原始静态求解器（分支定界 + 分段线性，**算法未做任何改动**），`validator.py` 校验解的可行性 |
| `dwta/` | `instance.py` 旧版动态算例解析；`dn_instance.py` **DN-WTA v1** 解析器（数据集规则.md：目标分时到达/距离演化/平台异质 `d0_ij`/延迟结算的纯数据层与动力学推导）；`wave_runner.py` 逐波调用求解器（子进程隔离，保持原求解器零侵入）；`simulator.py` 多波次蒙特卡洛仿真 |
| `llm/` | `llm_agent.py`：LLM 策略（模块 a/b/c/d 可组合），调用审计落盘 |
| `experiments/` | `gen_dynamic_data.py` 静态→动态算例生成；`gen_dn_data.py` DN-WTA v1 网格生成器（w/p 默认继承 `50x100x1.txt` 采样，池文件已清理、现自动回退同分布合成，可 `--pool` 指定）；`report.py` 结果报告 |

## 文档索引

- `项目整理规范.md` — 目录结构与重构红线
- `数据集规则.md` — DN-WTA v1 数据集与环境规范（本轮扩展依据）
- `DN-WTA_v1_数据集说明.md` — 新数据集结构/字段/动力学/清单与指标兼容性确认
- `DN-WTA_v1_下一步工作规划.md` — 统一多智能体环境与多算法对比的工作规划
- `实验综合报告.md` — 已完成实验的结论汇总
- `logs/LOGS_SUMMARY.md` — 保留日志的实验索引
- `output/README.md` — 实验结果文件夹索引与复现命令

> 历史文档说明：`实验报告_组会汇报版.md`（已并入 `实验综合报告.md`）与 `项目对比_wta-main_vs_branch01.md`（重构前版本对比，路径已过时）已于 2026-08-25 删除。
