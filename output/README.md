# output 目录说明（实验结果索引）

> **组织规则**：每次正式实验一个子文件夹（**单层**，文件夹内不再有子文件夹）。
> 文件夹内固定三类文件：`report.json`（结构化全量结果）、`report.md`（可读报告）、`llm_calls.jsonl`（LLM 调用审计，仅 LLM 实验）。

## 现有实验

| 文件夹 | 实验 | 说明 | 终端日志 |
|---|---|---|---|
| `e1_baseline/` | E1 | 纯 CPLEX 基线（无 LLM）， m=50 n=100 K=10, seeds 42-44 | `e1_baseline/e1_run.log`（随文件夹存放） |
| `e2_llm_ac/` | E2 | LLM 模块 `a+c`（`--llm-timeout 60`） | `../logs/e2_run.log` |
| `e3_llm_abc/` | E3 | LLM 模块 `a+b+c`（`--llm-timeout 60`） | `../logs/e3_run.log` |
| `e4_llm_abcd/` | E4 | LLM 全模块 `a+b+c+d`（`--llm-timeout 300`） | `../logs/e4_run.log` |

四个实验共用算例 `data/dyn_wta_50x100x3_K10_dist.txt`，蒙特卡洛种子 42/43/44，每波 CPLEX 时间上限 60s。

## 复现命令（在项目根目录执行）

```bash
# E1 基线
python main.py --instance data/dyn_wta_50x100x3_K10_dist.txt \
    --seeds 3 --seed-base 42 --timelimit 60 --policy base

# E2 / E3 / E4：把 <modules> 换成 a+c / a+b+c / a+b+c+d
python main.py --instance data/dyn_wta_50x100x3_K10_dist.txt \
    --seeds 3 --seed-base 42 --timelimit 60 \
    --policy llm --llm-modules <modules> --llm-model deepseek-v4-flash --llm-timeout <60|300>
```

## 新实验命名约定

`e<序号>_<简短标识>/`，如 `e5_llm_abd/`、`e6_scale_100/`；同一次运行的所有产物放同一文件夹，不要散落在 output 根目录。运行期间的 `tmp/` 波次中间文件由 `main.py` 正常结束时自动清理，无需手动保留。
