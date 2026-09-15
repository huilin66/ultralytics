# GIA / GCA / GIA+GCA / HO 实验结果

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED
- Version Label: exp_result_v1

本文档汇总当前远程服务器 `/localnvme/project/ultralytics` 中已经完成的
GIA、GCA/GNN、GIA+GCA 以及 HO 实验。主结果优先采用修正后的固定 Stage1、
Stage2=100、`w4=0.5` 协议；早期未匹配协议的结果只作为历史记录保留。

## 1. 指标与实验口径

- 检测指标：`mAP50` 为主指标，`mAP50-95` 为补充指标。
- 属性指标：`F1_macro(A)` 记作 `F1_attr@IoU0.5`。它是在 IoU≥0.5 的检测框匹配
  上统计属性混淆矩阵后计算的属性宏平均 F1，不是独立于检测框的 oracle-box F1。
- `F1_attr@IoU0.5` 与检测 mAP50 使用相同的 IoU=0.5 匹配语义，但 mAP50 还会
  沿置信度排序计算 AP，因此二者不是同一个数值。
- E2.14 的 GCA-only 重复实验包含 seed=1、2 两个新增 seed；因此该组报告
  `n=2`，不能写成完整三 seed 结果。
- E2.13、E2.15 均使用 seed=0、1、2；表中 `均值±样本标准差` 均在 seed 维度计算。
- E2.16 是对已训练的 GIA+Cross-GCN checkpoint 做 HO 推理切换，不重新训练。
  one-to-many 的日志只输出三位小数，因此相关均值和差值标为近似值。

## 2. 主比较结果

主表中的 GIA-only 使用 E2.15 的匹配 Stage2 control；GCA-only 使用 E2.14 的
Cross-GCN；GIA+GCA 使用 E2.13 的 Cross-GCN。这样可以清楚区分：

| 方案 | 实验来源 | n | Val mAP50 | Val F1_attr@0.5 | Test mAP50 | Test F1_attr@0.5 |
|---|---|---:|---:|---:|---:|---:|
| Baseline（无 GIA/GCA） | E1, `w4=0.5`, seed=0 | 1 | 0.6702 | 0.6841 | 0.6664 | 0.6457 |
| GIA only | E2.15 control | 3 | 0.6909 | 0.6827±0.0007 | 0.6042 | 0.6460±0.0063 |
| GCA only（Cross-GCN） | E2.14 | 2 | 0.6702 | 0.6803±0.0069 | 0.6664 | 0.6207±0.0253 |
| GIA+GCA（Cross-GCN） | E2.13 | 3 | 0.6907 | 0.6819±0.0010 | 0.6042 | 0.6552±0.0122 |
| GIA+GCA+HO（one-to-many） | E2.16 | 3 | ≈0.702 | ≈0.665±0.005 | ≈0.623 | ≈0.649±0.023 |

说明：E2.13/E2.15 的检测分支在 Stage2 中冻结，所以 GIA+GCA 与 GIA control
的 mAP 基本一致；GCA 的主要差异体现在属性分支。HO 则改变推理时采用的检测
分支，因此 mAP 发生变化。

## 3. GIA：E2.1 GIA-v2 position sweep

以下为 E2.1 的 Stage1 checkpoint 结果。`gia_v2_9` 在 Val mAP50、Val 属性 F1
和 Test 属性 F1 上均为该组最优，因此被选作后续 GIA Stage1 初始化。

| GIA 变体 | Val mAP50 | Val F1_attr@0.5 | Test mAP50 | Test F1_attr@0.5 |
|---|---:|---:|---:|---:|
| `gia_v2_5_7` | 0.6264 | 0.6424 | 0.6893 | 0.6570 |
| `gia_v2_6` | 0.6651 | 0.6195 | 0.6160 | 0.6139 |
| `gia_v2_7` | 0.6716 | 0.6327 | 0.6529 | 0.6775 |
| `gia_v2_8` | 0.6760 | 0.6091 | 0.5942 | 0.5982 |
| **`gia_v2_9`** | **0.6909** | **0.6821** | 0.6042 | **0.6781** |
| `gia_v2_10` | 0.6824 | 0.6529 | 0.6103 | 0.6436 |
| `gia_v2_13` | 0.6813 | 0.6255 | 0.6204 | 0.6296 |
| `gia_v2_16` | 0.6671 | 0.6433 | 0.6319 | 0.6297 |
| `gia_v2_19` | 0.6752 | 0.6059 | 0.5949 | 0.6301 |
| `gia_v2_22` | 0.6627 | 0.6502 | 0.6454 | 0.6456 |

选定 checkpoint：

```text
runs/experiments/E2_1_GIA_v2_position/
E2_1_GIA_v2_position_gia_v2_9_stage1_100_w4_0p5_seed_0/weights/best.pt
```

注意：上表是 E2.1 的 Stage1 checkpoint；用于公平的 GIA-only / GIA+GCA Stage2
比较时，应使用下面 E2.15 的 matched control，不能把两个阶段的数值混为同一结果。

## 4. GCA/GNN only：E2.14（不使用 GIA 初始化）

E2.14 从同一个普通 E1 Stage1 checkpoint 开始，仅训练 Stage2 属性/GNN 部分，
重复 seed=1、2。检测分支冻结，因此检测 mAP 与 baseline 保持不变。

| GCA/GNN 结构 | n | Val mAP50 | Val F1_attr@0.5 | Test mAP50 | Test F1_attr@0.5 |
|---|---:|---:|---:|---:|---:|
| Conditional-GCN margin residual | 2 | 0.6702 | 0.6799±0.0074 | 0.6664 | 0.6208±0.0254 |
| **Cross-GCN margin residual** | 2 | 0.6702 | 0.6803±0.0069 | 0.6664 | 0.6207±0.0253 |
| GraphSAGE adaptive residual | 2 | 0.6702 | **0.6845±0.0061** | 0.6664 | 0.6128±0.00005 |

该组没有观察到稳定的 Test F1 提升；它的作用主要是提供无 GIA 初始化的 GCA
对照，供后续和 E2.13 的 GIA+GCA 结果比较。

## 5. GIA+GCA：E2.13 与 E2.15 matched comparison

所有结构都从同一个 `gia_v2_9` Stage1 checkpoint 开始，Stage2=100，使用
seed=0、1、2。E2.15 是不加 GCA/GNN 的 GIA Stage2 control。

| 结构 | n | Val mAP50 | Val F1_attr@0.5 | Test mAP50 | Test F1_attr@0.5 | 相对 GIA control 的 Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| GIA only（E2.15 control） | 3 | 0.6909 | 0.6827±0.0007 | 0.6042 | 0.6460±0.0063 | — |
| GIA + Conditional-GCN | 3 | 0.6907 | 0.6824±0.0002 | 0.6042 | 0.6529±0.0036 | +0.0069 |
| GIA + **Cross-GCN** | 3 | 0.6907 | 0.6819±0.0010 | 0.6042 | 0.6552±0.0122 | +0.0093 |
| GIA + GraphSAGE adaptive | 3 | 0.6907 | **0.6835±0.0011** | 0.6042 | **0.6699±0.0151** | **+0.0239** |

从该 matched comparison 看，GraphSAGE adaptive 的三 seed 平均属性 F1 最高；
Cross-GCN 的 Test F1 也比 GIA control 高约 0.93 个百分点。当前 HO 实验选择
Cross-GCN，是因为 HO 对照是在该结构上实际执行的。

## 6. HO：GIA+Cross-GCN 的 native vs one-to-many

E2.16 对 E2.13 Cross-GCN 的三个 checkpoint 分别切换推理 head。native 数值来自
E2.13 的精确汇总；one-to-many 数值来自 E2.16 日志，日志只保留三位小数。

| 推理模式 | n | Val mAP50 | Val mAP50-95 | Val F1_attr@0.5 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native one-to-one | 3 | 0.6907 | 0.4548 | 0.6819±0.0010 | 0.6042 | 0.4328 | 0.6552±0.0122 |
| **One-to-many** | 3 | **≈0.702** | **≈0.459** | ≈0.665±0.005 | **≈0.623** | **≈0.450** | ≈0.649±0.023 |
| 差值（one-to-many − native） | — | **≈+0.0113** | **≈+0.0042** | ≈−0.0169 | **≈+0.0188** | **≈+0.0172** | ≈−0.0066 |

结论：one-to-many 明显提高检测 mAP50 和 mAP50-95；当前条件下，条件属性 F1
略有下降。由于属性 F1 只在 IoU≥0.5 的检测匹配上计算，这个下降不能直接解释
为属性 head 本身退化，也可能来自检测匹配集合变化。

### 6.1 联合指标的补充计算

若定义等权联合分数

```text
J = mAP50 × F1_attr@IoU0.5
```

则当前 Cross-GCN HO 结果约为：

| 推理模式 | Val J | Test J |
|---|---:|---:|
| Native one-to-one | 0.4710 | 0.3959 |
| One-to-many | 0.4668 | 0.4041 |

因此，等权 `J` 在 Val 上并不支持 one-to-many；选择 one-to-many 应明确表述为
“检测性能优先的 HO 选择”，并同时报告 mAP 与属性 F1。若使用加权几何分数，
权重必须在模型选择前根据任务目标预先固定，不能根据 Test 结果事后调节。

## 7. 历史结果（不纳入主比较）

早期 E2.2/E2.3 使用了未完全匹配的完整两阶段训练协议，保留用于追溯，但不与
E2.13/E2.14/E2.15 的 matched comparison 混合：

| 历史实验 | Val mAP50 | Val F1_attr@0.5 | Test mAP50 | Test F1_attr@0.5 | 备注 |
|---|---:|---:|---:|---:|---|
| E2.2 original GCA stage2 | 0.6595 | 0.6017 | 0.5988 | 0.5987 | 早期完整 GCA 结构 |
| E2.3 original GIA+GCA stage2 | 0.6789 | 0.6423 | 0.6132 | 0.5918 | 未使用后续 matched Stage1 protocol |

## 8. 原始结果位置

远程服务器项目根目录为 `/localnvme/project/ultralytics`：

```text
runs/experiments/E1_w4/summary.csv
runs/experiments/E2_1_GIA_v2_position/summary.csv
runs/experiments/E2_14_GCA_GNN_repeat_no_gia/summary.csv
runs/experiments/E2_13_GCA_GIA_test3_seed/summary.csv
runs/experiments/E2_15_GIA_stage2_control/summary.csv
runs/experiments/E2_16_HO_cross_gcn.log
runs/experiments/E2_16_HO_cross_gcn_test.log
```

E2.16 的 one-to-many 验证输出目录为：

```text
runs/experiments/E2_16_HO/
```

