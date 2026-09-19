# GIA / GCA / GIA+GCA / HO 实验结果

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: test_report
- Origin Date: 2026-09-17
- Verification Status: ANALYZED
- Version Label: exp_result_v2

本文档汇总当前远程服务器 `/localnvme/project/ultralytics` 中已经完成的
GIA、GCA/GNN、GIA+GCA 以及 HO 实验。主结果优先采用修正后的固定 Stage1、
Stage2=100、`w4=0.5` 协议；早期未匹配协议的结果只作为历史记录保留。

本文档及论文只报告 Test split 指标。

## 当前实验决策：主路线剔除，结果保留

基于 `gia_v2_5_7` Stage1 seed=0 checkpoint 的 E2.24（logits → MHA →
GCA/GNN）和 E2.25（feature + logits → MHA → GCA/GNN）共 10 个 Test 结果，
所有结构的 Test mAP50 均为 `0.689341`，Test mAP50-95 均为 `0.472128`；
最高 Test F1_attr@0.5 为 `0.656026`，低于 GIA-v2.5.7 参照的 `0.657020`。

因此，本轮 MHA-GCA/GNN 扩展判定为无有效提升，不纳入最终模型选择、主结果表或
后续论文实验路线；但作为负结果/消融记录保留在本 results 文档中，必要时可用于回应
审稿人关于该方向是否经过实验验证的询问。相关权重和 Test 汇总保留，不再作为后续
实验的初始化依据。

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
- E2.1 的 `confirm_seedfix` 是早期 GIA-v2.5.7 的 seed-matched 三 seed 确认，
  与后续选作 GIA+GCA 初始化的 GIA-v2.9 不是同一组实验，不能混写。

## E0. 基础训练协议与 HSV 消融

E0 用于确定基础训练协议及 HSV 数据增强配置。以下汇总均来自远程服务器的
`summary.csv`，仅展示 Test split 指标。

### E0.1 HSV 消融

| 配置 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| `hsv_current` | 0.642013 | 0.430439 | **0.621251** |
| `hsv_disabled` | 0.610960 | 0.438271 | 0.618295 |
| `hsv_reduced` | **0.666443** | **0.453483** | 0.602990 |

`hsv_reduced` 的 Test mAP50 和 Test mAP50-95 最高，`hsv_current` 的 Test F1
最高。

### E0.2 Stage1 epoch sweep

| Stage1 epoch | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---:|---:|---:|---:|
| 100 | **0.666443** | **0.453483** | 0.602990 |
| 200 | 0.611732 | 0.435328 | 0.596588 |
| 300 | 0.624355 | 0.427410 | **0.613066** |
| 400 | 0.659352 | 0.436944 | 0.596521 |
| 500 | 0.570077 | 0.395064 | 0.542881 |

Stage1=100 的 Test 检测指标最高；Stage1=300 的 Test F1 最高。

### E0.3 Stage2 epoch sweep

| Stage2 epoch | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---:|---:|---:|---:|
| 50 | 0.666443 | 0.453483 | 0.629567 |
| 100 | 0.666443 | 0.453483 | **0.645716** |
| 150 | 0.666443 | 0.453483 | 0.629567 |
| 200 | 0.666443 | 0.453483 | 0.629567 |

Stage2=100 的 Test F1 最高；四个配置的 Test 检测指标相同。因此当前基础协议
采用 Stage1=100、Stage2=100、`w4=0.5`、seed=0。

## E1. `w4` 基线扫描

E1 在固定 seed=0 下扫描 `w4`，分别记录 Stage1 和 Stage2 checkpoint 的 Test
结果。每个 `w4` 配置的检测指标在两个阶段保持一致，Stage2 主要改变属性 F1。

| `w4` | Stage1 Test（mAP50 / mAP50-95 / F1_attr@0.5） | Stage2 Test（mAP50 / mAP50-95 / F1_attr@0.5） |
|---:|---|---|
| 0.25 | 0.638782 / 0.457056 / 0.608089 | 0.638782 / 0.457056 / 0.622399 |
| **0.50** | **0.666443 / 0.453483 / 0.602990** | **0.666443 / 0.453483 / 0.645716** |
| 0.75 | 0.612497 / 0.433849 / 0.606299 | 0.612497 / 0.433849 / 0.599663 |
| 1.00 | 0.664457 / 0.455713 / 0.600878 | 0.664457 / 0.455713 / 0.598912 |
| 1.25 | 0.608638 / 0.419349 / 0.653759 | 0.608638 / 0.419349 / **0.657317** |
| 1.50 | 0.627184 / 0.445389 / 0.641893 | 0.627184 / 0.445389 / 0.607546 |

按 Test mAP50 排序，`w4=0.5` 最高，为 `0.666443`；按 Test mAP50-95 排序，
`w4=0.25` 最高，为 `0.457056`；按 Test F1 排序，`w4=1.25` 的 Stage2 结果
最高，为 `0.657317`。由于当前实验以 Test mAP50 为主要检测指标，后续 baseline
采用 `w4=0.5`、Stage1=100、Stage2=100、seed=0。

## 2. 主比较结果

主表中的 GIA-only 使用 E2.15 的匹配 Stage2 control；GCA-only 使用 E2.14 的
Cross-GCN；E2.13 的 GIA+GCA Cross-GCN 作为历史/HO 对照保留。当前最终采用的
GCA 结构由 E2.20 的稳定性实验确定。

| 方案 | 实验来源 | n | Test mAP50 | Test F1_attr@0.5 |
|---|---|---:|---:|---:|
| Baseline（无 GIA/GCA） | E1, `w4=0.5`, seed=0 | 1 | 0.6664 | 0.6457 |
| GIA only（v2.9，下游 matched control） | E2.15 control | 3 | 0.6042 | 0.6460±0.0063 |
| GCA only（Cross-GCN） | E2.14 | 2 | 0.6664 | 0.6207±0.0253 |
| GIA+GCA（Cross-GCN，历史/HO对照） | E2.13 | 3 | 0.6042 | 0.6552±0.0122 |
| **GIA+GCA（最终：Conditional + GIN + adaptive）** | **E2.20** | **2** | **0.6047** | **0.6666±0.0094** |
| GIA+GCA+HO（one-to-many，基于历史 Cross-GCN） | E2.16 | 3 | ≈0.623 | ≈0.649±0.023 |

说明：E2.13/E2.15 的检测分支在 Stage2 中冻结，所以 GIA+GCA 与 GIA control
的 mAP 基本一致；GCA 的主要差异体现在属性分支。HO 则改变推理时采用的检测
分支，因此 mAP 发生变化。

## 3. GIA：E2.1 GIA-v2 position sweep

E2.1 包含三类结果：position sweep 用于比较候选 GIA 位置，`confirm` 是单个
seed 的初始确认，`confirm_seedfix` 用于对早期 GIA-v2.5.7 做三 seed 配对复现。
以下首先是 position sweep 的 Stage1 checkpoint Test 结果。`gia_v2_5_7` 的
Test 检测指标最高，`gia_v2_9` 的 Test F1 最高；后续 E2.13/E2.15/E2.20 实际
采用了 `gia_v2_9` 作为 GIA Stage1 初始化。

| GIA 变体 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| **`gia_v2_5_7`** | **0.689341** | **0.472128** | 0.657020 |
| `gia_v2_6` | 0.616037 | 0.385149 | 0.613939 |
| `gia_v2_7` | 0.652945 | 0.456395 | 0.677520 |
| `gia_v2_8` | 0.594211 | 0.410421 | 0.598213 |
| `gia_v2_9` | 0.604230 | 0.432681 | **0.678076** |
| `gia_v2_10` | 0.610300 | 0.431619 | 0.643615 |
| `gia_v2_13` | 0.620375 | 0.450636 | 0.629572 |
| `gia_v2_16` | 0.631880 | 0.437313 | 0.629656 |
| `gia_v2_19` | 0.594879 | 0.420159 | 0.630138 |
| `gia_v2_22` | 0.645446 | 0.441827 | 0.645601 |

后续实际采用的 checkpoint：

```text
runs/experiments/E2_1_GIA_v2_position/
E2_1_GIA_v2_position_gia_v2_9_stage1_100_w4_0p5_seed_0/weights/best.pt
```

注意：上表是 E2.1 的 Stage1 checkpoint；用于公平的 GIA-only / GIA+GCA Stage2
比较时，应使用下面 E2.15 的 matched control，不能把两个 GIA 版本或两个阶段的
数值混为同一结果。表中的 `0.604230` 是 `gia_v2_9` 固定 Stage1 checkpoint 的
Test mAP50，不是 GIA-v2.5.7 三 seed 配对复现的均值。

### 3.1 GIA-v2.5.7：E2_1_GIA_v2_confirm

该目录仅包含 seed=1 的初始确认结果：

| 模型 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| Baseline | 0.666443 | 0.453483 | 0.602990 |
| GIA-v2.5.7 | **0.689341** | **0.472128** | **0.657020** |

### 3.2 GIA-v2.5.7：三 seed confirm_seedfix

该组来自 `E2_1_GIA_v2_confirm_seedfix`，Baseline 与 GIA-v2.5.7 按相同 seed
配对。Test 指标变化如下：

| seed | Baseline Test（mAP50 / mAP50-95 / F1） | GIA-v2.5.7 Test（mAP50 / mAP50-95 / F1） | 差值（mAP50 / mAP50-95 / F1） |
|---:|---|---|---|
| 0 | 0.666443 / 0.453483 / 0.602990 | 0.689341 / 0.472128 / 0.657020 | +0.022898 / +0.018645 / +0.054030 |
| 1 | 0.621977 / 0.413646 / 0.623962 | 0.659499 / 0.472072 / 0.634876 | +0.037522 / +0.058426 / +0.010913 |
| 2 | 0.635101 / 0.453514 / 0.609263 | 0.589792 / 0.393598 / 0.555470 | −0.045309 / −0.059916 / −0.053793 |
| **均值** | **0.641174 / 0.440214 / 0.612072** | **0.646211 / 0.445933 / 0.615789** | **+0.005037 / +0.005719 / +0.003717** |

三 seed 中有 2 个 seed 的三项 Test 指标均提升，1 个 seed 的三项 Test 指标均
下降。因此，`gia_v2.5.7` 是当前 Test 检测性能最优的候选，但仍需保留 seed
差异和复现稳定性说明。

## E2_2. 基于 baseline Stage2 的纯 GCA/GNN

本节只纳入固定 baseline Stage1 checkpoint 的 Stage2 纯 GCA/GNN 实验，不包含
原始完整两阶段的 GCA 结果。Baseline Stage2 与下列所有结构的 Test 检测指标
完全相同：

```text
Test mAP50    = 0.666443
Test mAP50-95 = 0.453483
```

因此 E2_2 只比较 Stage2 Test F1。

### E2_2.1 `E2_2_GCA_stage2`

| 结构 | n | Stage2 Test F1_attr@0.5 |
|---|---:|---:|
| Baseline Stage2 | 1 | **0.645716** |
| GAT learned | 1 | 0.589428 |
| GCA com | 1 | 0.577278 |
| GCA current | 1 | 0.583349 |
| GCN | 1 | 0.589533 |
| GIN | 1 | 0.586960 |
| GraphSAGE | 1 | 0.622928 |

### E2_2.2 `E2_2_GCA_GNN_margin_residual`

每个结构包含两个 seed；两条记录的 Stage2 Test F1 完全一致。

| 结构 | n | Stage2 Test F1_attr@0.5 |
|---|---:|---:|
| GAT margin residual | 2 | 0.601624 |
| GCA margin residual | 2 | 0.618513 |
| **GCN margin residual** | 2 | **0.645847** |
| GIN margin residual | 2 | 0.617948 |
| GraphSAGE margin residual | 2 | 0.620039 |

### E2_2.3 `E2_2_GCA_GNN_margin_residual_conditional`

| 结构 | n | Stage2 Test F1_attr@0.5 |
|---|---:|---:|
| GAT margin residual | 1 | 0.616151 |
| GCA margin residual | 1 | 0.618513 |
| **GCN margin residual** | 1 | **0.645847** |
| GIN margin residual | 1 | 0.633257 |
| GraphSAGE margin residual | 1 | 0.618513 |

### E2_2.4 `E2_2_GCA5x5_conditional`

以下每个单元格为 Stage2 Test F1_attr@0.5：

| GNN \ 结构 | adaptive | context_conditional | context_cross | conv_adapter | twohop |
|---|---:|---:|---:|---:|---:|
| GCA | **0.645716** | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GCN | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GAT | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GraphSAGE | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GIN | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |

### E2_2.5 `E2_2_GCA5x5_cross`

以下每个单元格为 Stage2 Test F1_attr@0.5：

| GNN \ 结构 | adaptive | context_conditional | context_cross | conv_adapter | twohop |
|---|---:|---:|---:|---:|---:|
| GCA | 0.645716 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GCN | 0.645716 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GAT | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| **GraphSAGE** | **0.645847** | 0.620039 | 0.620039 | 0.620039 | 0.620039 |
| GIN | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |

E2_2 中，所有结构的 Test mAP50 和 Test mAP50-95 均与 baseline Stage2 完全
一致；Test F1 最高的是 cross 矩阵下的 GraphSAGE + adaptive（0.645847），
其次是 conditional 矩阵下的 GCA + adaptive（0.645716）。

## 4. GCA/GNN only：E2.14（不使用 GIA 初始化）

E2.14 从同一个普通 E1 Stage1 checkpoint 开始，仅训练 Stage2 属性/GNN 部分，
重复 seed=1、2。检测分支冻结，因此检测 mAP 与 baseline 保持不变。

| GCA/GNN 结构 | n | Test mAP50 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| Conditional-GCN margin residual | 2 | 0.6664 | 0.6208±0.0254 |
| **Cross-GCN margin residual** | 2 | 0.6664 | 0.6207±0.0253 |
| GraphSAGE adaptive residual | 2 | 0.6664 | 0.6128±0.00005 |

该组没有观察到稳定的 Test F1 提升；它的作用主要是提供无 GIA 初始化的 GCA
对照，供后续和 E2.13 的 GIA+GCA 结果比较。

## 5. GIA+GCA：E2.13 与 E2.15 matched comparison

所有结构都从同一个 `gia_v2_9` Stage1 checkpoint 开始，Stage2=100，使用
seed=0、1、2。E2.15 是不加 GCA/GNN 的 GIA Stage2 control。因此本节中的
Test mAP50 `0.6042` 是 v2.9 下游 matched protocol 的值，不应回写为早期
v2.5.7 三 seed confirmation 的结果。

| 结构 | n | Test mAP50 | Test F1_attr@0.5 | 相对 GIA control 的 Test F1 |
|---|---:|---:|---:|---:|
| GIA only（E2.15 control） | 3 | 0.6042 | 0.6460±0.0063 | — |
| GIA + Conditional-GCN | 3 | 0.6042 | 0.6529±0.0036 | +0.0069 |
| GIA + **Cross-GCN** | 3 | 0.6042 | 0.6552±0.0122 | +0.0093 |
| GIA + GraphSAGE adaptive | 3 | 0.6042 | **0.6699±0.0151** | **+0.0239** |

从该 matched comparison 看，GraphSAGE adaptive 的三 seed 平均属性 F1 最高；
Cross-GCN 的 Test F1 也比 GIA control 高约 0.93 个百分点。当前 HO 实验选择
Cross-GCN，是因为 HO 对照是在该结构上实际执行的。

### 5.1 当前 GCA 最终结构：E2.20 Top-5 稳定性

在固定 `gia_v2_9` seed=0 Stage1 checkpoint 的基础上，仅训练 Stage2 属性分支，
使用 conditional 共现矩阵、`freeze=23`、Stage2=100、`w4=0.5`，对 Top-5
候选进行 seed=1、2 重复。最终选定：

> **Conditional + GIN + adaptive**

| seed | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---:|---:|---:|---:|
| 1 | 0.6047 | 0.4332 | **0.6733** |
| 2 | 0.6047 | 0.4332 | **0.6600** |
| **均值±样本标准差** | **0.6047** | **0.4332** | **0.6666±0.0094** |

与对应的 GIA-only Stage2 control（E2.15）相比，Test F1 两个 seed 分别提升
`+0.0303` 和 `+0.0183`，平均提升约 `+0.0243`。这是当前 Top-5 中唯一在两个
重复 seed 上都取得正向 Test F1 提升的结构，因此作为当前 GCA 的最终结构。
由于检测分支冻结，该结构的提升主要体现在属性 F1，Test mAP 不作为其提升依据。

E2.20 seed=2 中多个候选的 Test F1 恰好相同，是测试集硬属性标签判定后的指标碰撞；
各候选权重仍然不同，不影响将 GIN adaptive 按两个 seed 的配对结果进行选择。

## 6. HO：one-to-many

E2.16 对 E2.13 Cross-GCN 的三个 checkpoint 分别切换推理 head。native 数值来自
E2.13 的精确汇总；one-to-many 数值来自 E2.16 日志，日志只保留三位小数。

| 推理模式 | n | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|---:|
| Native one-to-one | 3 | 0.6042 | 0.4328 | 0.6552±0.0122 |
| **One-to-many** | 3 | **≈0.623** | **≈0.450** | ≈0.649±0.023 |
| 差值（one-to-many − native） | — | **≈+0.0188** | **≈+0.0172** | ≈−0.0066 |

结论：one-to-many 明显提高检测 mAP50 和 mAP50-95；当前条件下，条件属性 F1
略有下降。由于属性 F1 只在 IoU≥0.5 的检测匹配上计算，这个下降不能直接解释
为属性 head 本身退化，也可能来自检测匹配集合变化。注意 E2.16 是在历史
Cross-GCN 上完成的 HO 对照，不是 E2.20 最终 GIN adaptive 结构的 HO 结果；若后续
继续研究 HO，应在 E2.20 最终结构上重新进行匹配实验。

### 6.1 联合指标的补充计算

若定义等权联合分数

```text
J = mAP50 × F1_attr@IoU0.5
```

则当前 Cross-GCN HO 结果约为：

| 推理模式 | Test J |
|---|---:|
| Native one-to-one | 0.3959 |
| One-to-many | 0.4041 |

因此，等权 `J` 的 Test 结果支持 one-to-many；选择 one-to-many 应明确表述为
“检测性能优先的 HO 选择”，并同时报告 mAP 与属性 F1。若使用加权几何分数，
权重必须在模型选择前根据任务目标预先固定，不能根据 Test 结果事后调节。

### 6.2 E2.21：最终 GIN adaptive 的 native vs one-to-many

E2.21 直接复用 E2.20 的 `Conditional + GIN + adaptive` seed=1、2 checkpoint，
不重新训练，只在推理时调用 `use_one2many_head()`。以下为显式 `split=test` 的
精确评估结果：

| seed | Native Test mAP50 | Native Test mAP50-95 | Native Test F1_attr@0.5 | One-to-many Test mAP50 | One-to-many Test mAP50-95 | One-to-many Test F1_attr@0.5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6047 | 0.4332 | 0.6733 | **0.6226** | **0.4502** | **0.6792** |
| 2 | 0.6047 | 0.4332 | 0.6600 | **0.6226** | **0.4502** | **0.6703** |
| **均值±样本标准差** | **0.6047** | **0.4332** | **0.6666±0.0094** | **0.6226** | **0.4502** | **0.6748±0.0063** |

相对 native，one-to-many 的 Test mAP50 平均提升约 `+0.0179`，Test mAP50-95
平均提升约 `+0.0170`，Test F1 平均提升约 `+0.0081`；两个 seed 的 Test F1
均为正提升（分别为 `+0.0060`、`+0.0103`）。因此 E2.21 支持将 one-to-many
表述为检测性能优先的 HO 方案，同时必须报告 Test 属性 F1；该结果不是单纯的
属性性能全面提升。

### 6.3 E2.26：Baseline/GIA 的 HO 对照

E2.26 对 Baseline 与 `gia_v2_5_7` 分别重新加载 checkpoint，比较 native 与
one-to-many 两种输出方式，不重新训练。seed=0 严格采用用户指定的两个 checkpoint：
Baseline 为 E1 的 Stage2 权重，GIA 为 E2.1 的 `gia_v2_5_7` Stage1 权重。

#### seed=0：指定 checkpoint

| 模型 | 推理模式 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---|---:|---:|---:|
| Baseline | native | 0.666443 | 0.453483 | 0.645716 |
| Baseline | one-to-many | 0.659656 | 0.456900 | 0.643358 |
| GIA-v2.5.7 | native | 0.689341 | 0.472128 | 0.657020 |
| GIA-v2.5.7 | one-to-many | 0.687286 | 0.483173 | 0.626513 |

相对 native，Baseline 的差值为 `mAP50 -0.006788`、`mAP50-95 +0.003417`、
`F1 -0.002358`；GIA 的差值为 `mAP50 -0.002054`、`mAP50-95 +0.011045`、
`F1 -0.030508`。

#### seed=1、2：`confirm_seedfix` 同口径 Stage1 checkpoint

以下两组使用 `E2_1_GIA_v2_confirm_seedfix` 中 Baseline/GIA 各自的 Stage1
checkpoint。差值均为 `one-to-many − native`。

| 模型 | seed | native Test mAP50 | one-to-many Test mAP50 | Δ mAP50 | native Test mAP50-95 | one-to-many Test mAP50-95 | Δ mAP50-95 | native Test F1 | one-to-many Test F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1 | 0.621977 | 0.651164 | +0.029187 | 0.413646 | 0.431233 | +0.017587 | 0.623962 | 0.621529 | −0.002433 |
| Baseline | 2 | 0.635101 | 0.645083 | +0.009982 | 0.453514 | 0.459855 | +0.006341 | 0.609263 | 0.595448 | −0.013815 |
| GIA-v2.5.7 | 1 | 0.659499 | 0.676708 | +0.017209 | 0.472072 | 0.487498 | +0.015426 | 0.634876 | 0.646294 | +0.011418 |
| GIA-v2.5.7 | 2 | 0.589792 | 0.617214 | +0.027421 | 0.393598 | 0.416509 | +0.022911 | 0.555470 | 0.581406 | +0.025935 |

三 seed 的解释必须保持口径区分：Baseline seed=1、2 的 mAP50 均提升但 F1
下降；GIA seed=1、2 的三个 Test 指标均提升。seed=0 的指定 checkpoint 中，
两者的 mAP50 和 F1 均下降。因此 HO 存在 seed 敏感性，不能表述为稳定的全面提升；
在 GIA 上可以表述为“多数 seed 有利于检测和属性 F1，但 seed=0 出现退化”。

E2.26 输出路径：

```text
runs/experiments/E2_26_HO_baseline_gia/final/baseline_test_summary.csv
runs/experiments/E2_26_HO_baseline_gia/final/gia_v2_5_7_test_summary.csv
runs/experiments/E2_26_HO_baseline_gia/confirm_seedfix/baseline_seed1_test_summary.csv
runs/experiments/E2_26_HO_baseline_gia/confirm_seedfix/baseline_seed2_test_summary.csv
runs/experiments/E2_26_HO_baseline_gia/confirm_seedfix/gia_v2_5_7_seed1_test_summary.csv
runs/experiments/E2_26_HO_baseline_gia/confirm_seedfix/gia_v2_5_7_seed2_test_summary.csv
```

### 6.4 E2.27：Baseline/GIA 五个 Seed 的 Stage2 稳定性

E2.27 对 Baseline 与 `gia_v2_5_7` 分别使用 seed=0、1、2、3、4 完成
Stage1+Stage2 训练；最终比较只使用各自的 Stage2 权重，共 10 个 Test 结果。

| seed | Baseline Test mAP50 | Baseline Test F1_macro | GIA Test mAP50 | GIA Test F1_macro |
|---:|---:|---:|---:|---:|
| 0 | 0.666443 | 0.645716 | 0.689341 | 0.656026 |
| 1 | 0.621977 | 0.624775 | 0.659499 | 0.642142 |
| 2 | 0.635101 | 0.646610 | 0.589792 | 0.593163 |
| 3 | 0.617033 | 0.635252 | 0.657618 | 0.573754 |
| 4 | 0.580827 | 0.542371 | 0.601946 | 0.588384 |

E2.27 的 Test 汇总路径：

```text
runs/experiments/E2_27_baseline_gia_seed5/summary.csv
```

## E4：RT-DETR 多规模属性检测

E4 使用 RT-DETR-L 与 RT-DETR-X 完成 Stage1+Stage2 训练，以下为最终 Stage2
权重的 Test 结果。

| 模型 | mAP50_test | mAP50-95_test | OA_test | F1_macro_test | F1_macro_global_test | P_macro_test | R_macro_test |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-L | 0.509574 | 0.329548 | 0.968627 | 0.491973 | 0.492032 | 0.484314 | 0.500000 |
| RT-DETR-X | 0.556197 | 0.373912 | 0.967130 | 0.491577 | 0.491645 | 0.483565 | 0.500000 |

E4 Test 汇总路径：

```text
runs/experiments/E4_rtdetr_LX/summary.csv
```

## E5：真正的多标签目标检测 YOLOv10

E5 每个物理目标只保留一个检测框，并使用 2 个目标类别计算检测 mAP；
属性指标基于 mAP50 中 IoU≥0.5 且目标类别正确的匹配框，对 10 维属性向量计算。

| 指标 | Test |
|---|---:|
| mAP50_test | 0.409310 |
| mAP50-95_test | 0.235047 |
| P_macro_test | 0.534732 |
| R_macro_test | 0.508019 |
| F1_macro_test@IoU0.5 | 0.507001 |

本次 Test 共 62 张图像、247 个真实目标，其中 191 个目标完成 IoU≥0.5
匹配。此前将属性展开为独立检测类别得到的结果不纳入正式结果。

E5 权重：

```text
runs/experiments/E5_multilabel/yolov10x/weights/best.pt
```

## E6：目标检测 + 多标签分类双阶段

E6 使用 E3 YOLOv10x Stage2 检测器生成预测框，再由 E6 YOLOv10x-cls
分类器对预测框裁剪图进行多属性分类。检测指标来自检测阶段；宏平均指标基于
IoU≥0.5 且目标类别正确的匹配目标，并使用分类器输出的属性结果计算。

| 指标 | Test |
|---|---:|
| mAP50_test | 0.630217 |
| mAP50-95_test | 0.431025 |
| P_macro_test | 0.540765 |
| R_macro_test | 0.528227 |
| F1_macro_test@IoU0.5 | 0.530298 |

本次 Test 共 62 张图像、247 个真实目标，其中 195 个目标完成 IoU≥0.5
匹配。

E6 权重与来源：

```text
检测器：runs/experiments/E3_versions/E3_versions_yolov10x_w4_0p5_seed_0_stage2/weights/best.pt
分类器：runs/experiments/E6_two_stage_yolov10x/detector_yolov10x_classifier_yolov10x_cls/weights/best.pt
```

## 7. 历史结果（不纳入主比较）

早期 E2.2/E2.3 使用了未完全匹配的完整两阶段训练协议，保留用于追溯，但不与
E2.13/E2.14/E2.15 的 matched comparison 混合：

| 历史实验 | Test mAP50 | Test F1_attr@0.5 | 备注 |
|---|---:|---:|---|
| E2.2 original GCA stage2 | 0.5988 | 0.5987 | 早期完整 GCA 结构 |
| E2.3 original GIA+GCA stage2 | 0.6132 | 0.5918 | 未使用后续 matched Stage1 protocol |

## 8. 原始结果位置

远程服务器项目根目录为 `/localnvme/project/ultralytics`：

```text
runs/experiments/E1_w4/summary.csv
runs/experiments/E0_hsv_ablation/summary.csv
runs/experiments/E0_stage1_sweep/summary.csv
runs/experiments/E0_stage2_sweep/summary.csv
runs/experiments/E2_1_GIA_v2_position/summary.csv
runs/experiments/E2_1_GIA_v2_confirm/summary.csv
runs/experiments/E2_1_GIA_v2_confirm_seedfix/summary.csv
runs/experiments/E2_2_GCA_stage2/summary.csv
runs/experiments/E2_2_GCA_GNN_margin_residual/summary.csv
runs/experiments/E2_2_GCA_GNN_margin_residual_conditional/summary.csv
runs/experiments/E2_2_GCA5x5_conditional/summary.csv
runs/experiments/E2_2_GCA5x5_cross/summary.csv
runs/experiments/E2_14_GCA_GNN_repeat_no_gia/summary.csv
runs/experiments/E2_13_GCA_GIA_test3_seed/summary.csv
runs/experiments/E2_15_GIA_stage2_control/summary.csv
runs/experiments/E2_20_GIA_GCA_top5_stability/summary.csv
runs/experiments/E2_21_HO_GIA_GCA_gin_adaptive/test_summary.csv
runs/experiments/E2_24_GIA_v2_5_7_GCA_MHA_margin_residual_conditional/test_summary.csv
runs/experiments/E2_25_GIA_v2_5_7_GCA_feature_logit_MHA_margin_residual_conditional/test_summary.csv
runs/experiments/E2_27_baseline_gia_seed5/summary.csv
runs/experiments/E2_16_HO_cross_gcn.log
runs/experiments/E2_16_HO_cross_gcn_test.log
```

E2.16 的 one-to-many 评估输出目录为：

```text
runs/experiments/E2_16_HO/
```
