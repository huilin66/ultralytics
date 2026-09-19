# GIA / GCA / GIA+GCA / HO 实验结果

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: test_report
- Origin Date: 2026-09-17
- Verification Status: ANALYZED
- Version Label: exp_result_v3

本文档只报告 Test split 指标。主结果按当前实验路线组织；旧版 GCA、GIA+GCA
和相关 HO 结果统一放在文末 102.x 历史备份区，不参与新版主比较。

## 1. 指标与实验口径

- 检测指标：mAP50 为主指标，mAP50-95 为补充指标。
- 属性指标：F1_macro(A) 记作 F1_attr@IoU0.5。它是在 IoU≥0.5 的检测框匹配
  上统计属性混淆矩阵后计算的属性宏平均 F1，不是独立于检测框的 oracle-box F1。
- F1_attr@IoU0.5 与检测 mAP50 使用相同的 IoU=0.5 匹配语义，但 mAP50 还会
  沿置信度排序计算 AP，因此二者不是同一个数值。

## E0. 基础训练协议与 HSV 消融

E0 用于确定基础训练协议及 HSV 数据增强配置。以下汇总均来自远程服务器的
summary.csv，仅展示 Test split 指标。

### E0.1 HSV 消融

| 配置 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| hsv_current | 0.642013 | 0.430439 | **0.621251** |
| hsv_disabled | 0.610960 | 0.438271 | 0.618295 |
| hsv_reduced | **0.666443** | **0.453483** | 0.602990 |

hsv_reduced 的 Test mAP50 和 Test mAP50-95 最高，hsv_current 的 Test F1
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

当前基础协议采用 Stage1=100、Stage2=100、w4=0.5、seed=0。

## E1. w4 基线扫描

E1 在固定 seed=0 下扫描 w4，分别记录 Stage1 和 Stage2 checkpoint 的 Test
结果。

| w4 | Stage1 Test（mAP50 / mAP50-95 / F1_attr@0.5） | Stage2 Test（mAP50 / mAP50-95 / F1_attr@0.5） |
|---:|---|---|
| 0.25 | 0.638782 / 0.457056 / 0.608089 | 0.638782 / 0.457056 / 0.622399 |
| **0.50** | **0.666443 / 0.453483 / 0.602990** | **0.666443 / 0.453483 / 0.645716** |
| 0.75 | 0.612497 / 0.433849 / 0.606299 | 0.612497 / 0.433849 / 0.599663 |
| 1.00 | 0.664457 / 0.455713 / 0.600878 | 0.664457 / 0.455713 / 0.598912 |
| 1.25 | 0.608638 / 0.419349 / 0.653759 | 0.608638 / 0.419349 / **0.657317** |
| 1.50 | 0.627184 / 0.445389 / 0.641893 | 0.627184 / 0.445389 / 0.607546 |

后续 baseline 采用 w4=0.5、Stage1=100、Stage2=100、seed=0。

## 2. GIA/Baseline 消融与稳定性

### 2.0 Baseline 五个 Seed 的 Stage2 稳定性（E2.27 前半部分）

E2.27 对 baseline 使用 seed=0、1、2、3、4 完成 Stage1+Stage2 训练；以下仅
比较各自 Stage2 权重。

| seed | Test mAP50 | Test mAP50-95 | OA_test | F1_macro_test | F1_macro_global_test | P_macro_test | R_macro_test |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.666443 | 0.453483 | 0.971503 | 0.645716 | 0.714910 | 0.699332 | 0.624985 |
| 1 | 0.621977 | 0.413646 | 0.970792 | 0.624775 | 0.706099 | 0.638335 | 0.618800 |
| 2 | 0.635101 | 0.453514 | 0.975676 | 0.646610 | 0.708976 | 0.696141 | 0.637032 |
| 3 | 0.617033 | 0.421210 | 0.970526 | 0.635252 | 0.694572 | 0.676286 | 0.616638 |
| 4 | 0.580827 | 0.373583 | 0.964286 | 0.542371 | 0.574209 | 0.544479 | 0.543974 |

### 2.1 GIA 增加实验与稳定性

#### E2.1 GIA-v2 position sweep

以下为 E2.1 各 GIA 变体 Stage1 checkpoint 的 Test 结果。

| GIA 变体 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| **gia_v2_5_7** | **0.689341** | **0.472128** | 0.657020 |
| gia_v2_6 | 0.616037 | 0.385149 | 0.613939 |
| gia_v2_7 | 0.652945 | 0.456395 | 0.677520 |
| gia_v2_8 | 0.594211 | 0.410421 | 0.598213 |
| gia_v2_9 | 0.604230 | 0.432681 | **0.678076** |
| gia_v2_10 | 0.610300 | 0.431619 | 0.643615 |
| gia_v2_13 | 0.620375 | 0.450636 | 0.629572 |
| gia_v2_16 | 0.631880 | 0.437313 | 0.629656 |
| gia_v2_19 | 0.594879 | 0.420159 | 0.630138 |
| gia_v2_22 | 0.645446 | 0.441827 | 0.645601 |

当前 GIA 结构和后续新增 GCA 实验的初始化基准为 gia_v2_5_7 的 Stage1 权重。

#### E2.27 GIA-v2.5.7 五个 Seed 的 Stage2 稳定性（后半部分）

| seed | Test mAP50 | Test mAP50-95 | OA_test | F1_macro_test | F1_macro_global_test | P_macro_test | R_macro_test |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.689341 | 0.472128 | 0.973333 | 0.656026 | 0.738252 | 0.691394 | 0.641018 |
| 1 | 0.659499 | 0.472072 | 0.973224 | 0.642142 | 0.723905 | 0.646559 | 0.643997 |
| 2 | 0.589792 | 0.393598 | 0.969444 | 0.593163 | 0.655449 | 0.648032 | 0.583081 |
| 3 | 0.657618 | 0.468003 | 0.967553 | 0.573754 | 0.632877 | 0.607304 | 0.562562 |
| 4 | 0.601946 | 0.424074 | 0.968984 | 0.588384 | 0.654854 | 0.619310 | 0.574664 |

E2.27 Test 汇总路径：

    runs/experiments/E2_27_baseline_gia_seed5/summary.csv

### 2.2 GCA 增加实验与稳定性

当前留白。新版 GCA 结果待 E2.28 的五 seed、cross/conditional 实验完成后再写入。

### 2.3 HO 增加实验

当前留白，等待新版 baseline 与 GIA 对照实验。

### 2.4 GIA+HO

当前留白，等待新版 GIA+HO 对照实验。

## 3. 各模型及尺寸对比

当前留白。待 E3 的各模型、各尺寸 Test 结果统一统计后写入。

## 4. 多标签目标检测与目标检测+多标签分类双阶段

### E4：RT-DETR 多规模属性检测

E4 使用 RT-DETR-L 与 RT-DETR-X 完成 Stage1+Stage2 训练。

| 模型 | mAP50_test | mAP50-95_test | OA_test | F1_macro_test | F1_macro_global_test | P_macro_test | R_macro_test |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-L | 0.509574 | 0.329548 | 0.968627 | 0.491973 | 0.492032 | 0.484314 | 0.500000 |
| RT-DETR-X | 0.556197 | 0.373912 | 0.967130 | 0.491577 | 0.491645 | 0.483565 | 0.500000 |

### E5：真正的多标签目标检测 YOLOv10

E5 每个物理目标只保留一个检测框，并使用 2 个目标类别计算检测 mAP；
属性指标基于 mAP50 中 IoU≥0.5 且目标类别正确的匹配框，对 10 维属性向量计算。

| 指标 | Test |
|---|---:|
| mAP50_test | 0.409310 |
| mAP50-95_test | 0.235047 |
| P_macro_test | 0.534732 |
| R_macro_test | 0.508019 |
| F1_macro_test@IoU0.5 | 0.507001 |

### E6：目标检测 + 多标签分类双阶段

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

## 5. 鲁棒性实验

当前留白，待后续鲁棒性实验完成后补充。

## 100. 原始结果位置

远程服务器项目根目录为 /localnvme/project/ultralytics：

    runs/experiments/E0_hsv_ablation/summary.csv
    runs/experiments/E0_stage1_sweep/summary.csv
    runs/experiments/E0_stage2_sweep/summary.csv
    runs/experiments/E1_w4/summary.csv
    runs/experiments/E2_1_GIA_v2_position/summary.csv
    runs/experiments/E2_27_baseline_gia_seed5/summary.csv
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_conditional/
    runs/experiments/E4_rtdetr_LX/summary.csv
    runs/experiments/E5_multilabel/yolov10x/weights/best.pt
    runs/experiments/E6_two_stage_yolov10x/

## 100.1 结果文件索引

E2.1 confirm 与 confirm_seedfix 的原始文件：

    runs/experiments/E2_1_GIA_v2_confirm/summary.csv
    runs/experiments/E2_1_GIA_v2_confirm_seedfix/summary.csv

旧版 GCA/GIA+GCA 文件：

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
    runs/experiments/E2_16_HO_cross_gcn.log
    runs/experiments/E2_16_HO_cross_gcn_test.log

E2.26 文件：

    runs/experiments/E2_26_HO_baseline_gia/

E2.27 文件：

    runs/experiments/E2_27_baseline_gia_seed5/summary.csv

## 102.1 旧版 GIA 确认与配对复现

以下结果为旧版 GIA 确认实验，已被 E2.27 的五 seed 稳定性实验替代，仅作备份。

### E2_1_GIA_v2_confirm

| 模型 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| Baseline | 0.666443 | 0.453483 | 0.602990 |
| GIA-v2.5.7 | **0.689341** | **0.472128** | **0.657020** |

### E2_1_GIA_v2_confirm_seedfix

| seed | Baseline Test（mAP50 / mAP50-95 / F1） | GIA-v2.5.7 Test（mAP50 / mAP50-95 / F1） | 差值（mAP50 / mAP50-95 / F1） |
|---:|---|---|---|
| 0 | 0.666443 / 0.453483 / 0.602990 | 0.689341 / 0.472128 / 0.657020 | +0.022898 / +0.018645 / +0.054030 |
| 1 | 0.621977 / 0.413646 / 0.623962 | 0.659499 / 0.472072 / 0.634876 | +0.037522 / +0.058426 / +0.010913 |
| 2 | 0.635101 / 0.453514 / 0.609263 | 0.589792 / 0.393598 / 0.555470 | −0.045309 / −0.059916 / −0.053793 |
| **均值** | **0.641174 / 0.440214 / 0.612072** | **0.646211 / 0.445933 / 0.615789** | **+0.005037 / +0.005719 / +0.003717** |

## 102.2 旧版 GCA/GNN、GIA+GCA 及相关 HO 结果

本节保留旧版 GCA 相关数据，不参与新版 2.2 的主结果。

### E2_2：基于 baseline Stage2 的纯 GCA/GNN

旧版 E2_2 中所有结构的 Test mAP50=0.666443、Test mAP50-95=0.453483，
因此只比较 Test F1。

#### E2_2_GCA_stage2

| 结构 | n | Test F1_attr@0.5 |
|---|---:|---:|
| Baseline Stage2 | 1 | **0.645716** |
| GAT learned | 1 | 0.589428 |
| GCA com | 1 | 0.577278 |
| GCA current | 1 | 0.583349 |
| GCN | 1 | 0.589533 |
| GIN | 1 | 0.586960 |
| GraphSAGE | 1 | 0.622928 |

#### E2_2_GCA_GNN_margin_residual

| 结构 | n | Test F1_attr@0.5 |
|---|---:|---:|
| GAT margin residual | 2 | 0.601624 |
| GCA margin residual | 2 | 0.618513 |
| **GCN margin residual** | 2 | **0.645847** |
| GIN margin residual | 2 | 0.617948 |
| GraphSAGE margin residual | 2 | 0.620039 |

#### E2_2_GCA_GNN_margin_residual_conditional

| 结构 | n | Test F1_attr@0.5 |
|---|---:|---:|
| GAT margin residual | 1 | 0.616151 |
| GCA margin residual | 1 | 0.618513 |
| **GCN margin residual** | 1 | **0.645847** |
| GIN margin residual | 1 | 0.633257 |
| GraphSAGE margin residual | 1 | 0.618513 |

#### E2_2_GCA5x5_conditional

| GNN / 结构 | adaptive | context_conditional | context_cross | conv_adapter | twohop |
|---|---:|---:|---:|---:|---:|
| GCA | **0.645716** | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GCN | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GAT | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GraphSAGE | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GIN | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |

#### E2_2_GCA5x5_cross

| GNN / 结构 | adaptive | context_conditional | context_cross | conv_adapter | twohop |
|---|---:|---:|---:|---:|---:|
| GCA | 0.645716 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GCN | 0.645716 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| GAT | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |
| **GraphSAGE** | **0.645847** | 0.620039 | 0.620039 | 0.620039 | 0.620039 |
| GIN | 0.629567 | 0.627916 | 0.627916 | 0.627916 | 0.627916 |

### E2.14：不使用 GIA 初始化的 GCA/GNN

| GCA/GNN 结构 | n | Test mAP50 | Test F1_attr@0.5 |
|---|---:|---:|---:|
| Conditional-GCN margin residual | 2 | 0.6664 | 0.6208±0.0254 |
| **Cross-GCN margin residual** | 2 | 0.6664 | 0.6207±0.0253 |
| GraphSAGE adaptive residual | 2 | 0.6664 | 0.6128±0.00005 |

### E2.13、E2.15：旧版 GIA+GCA matched comparison

| 结构 | n | Test mAP50 | Test F1_attr@0.5 | 相对 GIA control 的 Test F1 |
|---|---:|---:|---:|---:|
| GIA only（E2.15 control） | 3 | 0.6042 | 0.6460±0.0063 | — |
| GIA + Conditional-GCN | 3 | 0.6042 | 0.6529±0.0036 | +0.0069 |
| GIA + **Cross-GCN** | 3 | 0.6042 | 0.6552±0.0122 | +0.0093 |
| GIA + GraphSAGE adaptive | 3 | 0.6042 | **0.6699±0.0151** | **+0.0239** |

### E2.20：旧版 GIA+GCA Top-5 稳定性

| seed | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---:|---:|---:|---:|
| 1 | 0.6047 | 0.4332 | **0.6733** |
| 2 | 0.6047 | 0.4332 | **0.6600** |
| **均值±样本标准差** | **0.6047** | **0.4332** | **0.6666±0.0094** |

旧版最终结构记录为 Conditional + GIN + adaptive；该结果不作为新版 GCA
稳定性主结果。

### E2.16、E2.21：旧版 GIA+GCA 的 HO 结果

#### E2.16：Cross-GCN

| 推理模式 | n | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---:|---:|---:|---:|
| Native one-to-one | 3 | 0.6042 | 0.4328 | 0.6552±0.0122 |
| One-to-many | 3 | ≈0.623 | ≈0.450 | ≈0.649±0.023 |
| 差值 | — | ≈+0.0188 | ≈+0.0172 | ≈−0.0066 |

#### E2.21：Conditional + GIN + adaptive

| seed | Native Test mAP50 | Native Test mAP50-95 | Native Test F1_attr@0.5 | One-to-many Test mAP50 | One-to-many Test mAP50-95 | One-to-many Test F1_attr@0.5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6047 | 0.4332 | 0.6733 | **0.6226** | **0.4502** | **0.6792** |
| 2 | 0.6047 | 0.4332 | 0.6600 | **0.6226** | **0.4502** | **0.6703** |
| **均值±样本标准差** | **0.6047** | **0.4332** | **0.6666±0.0094** | **0.6226** | **0.4502** | **0.6748±0.0063** |

### E2.24、E2.25：MHA-GCA/GNN 负结果

基于 gia_v2_5_7 Stage1 seed=0 checkpoint 的两组 MHA-GCA/GNN 扩展共 10 个
Test 结果，所有结构的 Test mAP50 为 0.689341，Test mAP50-95 为 0.472128；
最高 Test F1_attr@0.5 为 0.656026，低于 GIA-v2.5.7 参照值 0.657020。

该方向不纳入新版模型选择和主结果，仅保留为旧版负结果记录。

### 其它旧版 GCA 结果

| 历史实验 | Test mAP50 | Test F1_attr@0.5 | 备注 |
|---|---:|---:|---|
| E2.2 original GCA stage2 | 0.5988 | 0.5987 | 早期完整 GCA 结构 |
| E2.3 original GIA+GCA stage2 | 0.6132 | 0.5918 | 未使用后续 matched Stage1 protocol |

## 102.3 其它旧版 HO 对照

### E2.26：Baseline/GIA 的旧版 HO 对照

#### seed=0：指定 checkpoint

| 模型 | 推理模式 | Test mAP50 | Test mAP50-95 | Test F1_attr@0.5 |
|---|---|---:|---:|---:|
| Baseline | native | 0.666443 | 0.453483 | 0.645716 |
| Baseline | one-to-many | 0.659656 | 0.456900 | 0.643358 |
| GIA-v2.5.7 | native | 0.689341 | 0.472128 | 0.657020 |
| GIA-v2.5.7 | one-to-many | 0.687286 | 0.483173 | 0.626513 |

#### seed=1、2：confirm_seedfix 同口径 Stage1 checkpoint

| 模型 | seed | native Test mAP50 | one-to-many Test mAP50 | Δ mAP50 | native Test mAP50-95 | one-to-many Test mAP50-95 | Δ mAP50-95 | native Test F1 | one-to-many Test F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1 | 0.621977 | 0.651164 | +0.029187 | 0.413646 | 0.431233 | +0.017587 | 0.623962 | 0.621529 | −0.002433 |
| Baseline | 2 | 0.635101 | 0.645083 | +0.009982 | 0.453514 | 0.459855 | +0.006341 | 0.609263 | 0.595448 | −0.013815 |
| GIA-v2.5.7 | 1 | 0.659499 | 0.676708 | +0.017209 | 0.472072 | 0.487498 | +0.015426 | 0.634876 | 0.646294 | +0.011418 |
| GIA-v2.5.7 | 2 | 0.589792 | 0.617214 | +0.027421 | 0.393598 | 0.416509 | +0.022911 | 0.555470 | 0.581406 | +0.025935 |
