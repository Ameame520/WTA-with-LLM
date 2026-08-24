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
├── data/              # 算例：静态 wta_*.txt 与动态 dyn_wta_*.txt
├── logs/              # 正式实验终端日志（索引见 logs/LOGS_SUMMARY.md）
└── output/            # 实验结果，每次实验一个子文件夹（索引见 output/README.md）
```

一个顶层文件夹 = 一个模块；文件夹内不再嵌套子文件夹（output 的实验文件夹、data 除外），详见 `项目整理规范.md`。

## 快速开始

所有命令**在项目根目录**执行；Python 环境需安装 CPLEX（默认解释器 `/opt/anaconda3/envs/wta/bin/python`）。

```bash
# 生成动态算例（静态算例 → K 波次动态算例）
python experiments/gen_dynamic_data.py --input data/wta_50x100x1.txt --waves 10 --seed 7

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
| `dwta/` | `instance.py` 算例解析；`wave_runner.py` 逐波调用求解器（子进程隔离，保持原求解器零侵入）；`simulator.py` 多波次蒙特卡洛仿真 |
| `llm/` | `llm_agent.py`：LLM 策略（模块 a/b/c/d 可组合），调用审计落盘 |
| `experiments/` | `gen_dynamic_data.py` 静态→动态算例生成；`report.py` 结果报告 |

## 文档索引

- `项目整理规范.md` — 目录结构与重构红线
- `实验综合报告.md` — 已完成实验的结论汇总
- `logs/LOGS_SUMMARY.md` — 保留日志的实验索引
- `output/README.md` — 实验结果文件夹索引与复现命令
- `实验报告_组会汇报版.md`、`项目对比_wta-main_vs_branch01.md` — 汇报与版本对比材料
