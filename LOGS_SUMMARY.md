# logs 目录总结（日志索引）

> 本目录只保留**正式实验**的终端运行日志。整理日期：2026-08-22。
> 更详细的结构化结果见 `output/<实验文件夹>/report.json`；跨实验对比见主目录 `实验综合报告.md`。

## 通用实验设置（E1–E4 相同）

| 项 | 值 |
|---|---|
| 动态算例 | `data/dyn_wta_50x100x3_K10_dist.txt`（m=50 武器， n=100 目标， mu=3 每武器弹药基数， K=10 波， 目标按波均衡到达） |
| 蒙特卡洛 | 3 个种子：42 / 43 / 44（`--seeds 3 --seed-base 42`） |
| 求解器 | 每波 CPLEX 子进程， `--timelimit 60`（每波 60s， 单种子全程 600s） |
| LLM 模型 | deepseek-v4-flash（E2–E4） |

## 保留的日志

| 日志文件 | 实验 | 关键配置 | 结果摘要 |
|---|---|---|---|
| `e2_run.log` | **E2** LLM 模块 `a+c` | `--policy llm --llm-modules a+c --llm-timeout 60` | 3 种子 leak value 全 0， leak rate 0.000000 |
| `e3_run.log` | **E3** LLM 模块 `a+b+c` | `--policy llm --llm-modules a+b+c --llm-timeout 60` | 同上，全 0 |
| `e4_run.log` | **E4** LLM 模块 `a+b+c+d`（全模块） | `--policy llm --llm-modules a+b+c+d --llm-timeout 300` | 同上，全 0 |
| *(另* `output/e1_baseline/e1_run.log` *\*)* | **E1** 纯 CPLEX 基线 | `--policy base`（无 LLM） | 同上，全 0（与 E1 报告同存于其实验文件夹） |

**注**：四个实验在当前算例规模下每波 60s 内均能达到泄漏值为 0 的解，因此 leak 指标无区分度；对比需看 `report.json` 中的逐波统计（分配结构、求解耗时、LLM 调用情况等）。LLM 调用的完整审计（提示词/回复）分别存于各实验文件夹的 `llm_calls.jsonl`。

## 实验递进关系

```
E1 基线(纯CPLEX) → E2 +模块a+c → E3 +模块b(a+b+c) → E4 +模块d(a+b+c+d, LLM超时放宽到300s)
```

## 已删除的日志（2026-08-22 清理）

- `dwta_20260816_*.log`（9 个）+ `results.csv`：0816 早期静态基准调试记录
- `dwta_20260820_233939/234547/235347.log`：0820 demo 调试与 E1 首跑（内容已被正式运行覆盖）
- `dwta_20260821_000752~003302.log`（7 个）+ `llm_calls_smoke.jsonl`：0821 K1 算例冒烟测试与 E4 中断前版本
- `dwta_20260822_194848.log`：0822 目录重构后的链路冒烟验证（2026-08-25 清理）
