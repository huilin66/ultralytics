import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
pd.set_option("display.precision", 4)

def load_confusion_matrix(file_path, risk_name):
    """加载混淆矩阵CSV文件"""
    # 检查文件是否存在
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取CSV文件，第一列作为索引
    cm_df = pd.read_csv(file_path, index_col=0)
    
    # 创建用于显示的DataFrame副本
    display_df = cm_df.copy()
    
    # 修改索引和列名，添加标识
    display_df.index = [f"pred: {cls}" for cls in display_df.index]
    display_df.columns = [f"label: {cls}" for cls in display_df.columns]
    
    # 计算各level的总标签数
    label_totals = cm_df.sum(axis=0)
    display_df.loc['label:total'] = label_totals.values 
    pred_totals = display_df.sum(axis=1)
    display_df['pred:total'] = pred_totals.values

    display_df = display_df.astype(int)
    # 打印混淆矩阵（含总数，带网格）
    print(f"{' ':20} {risk_name} confusion matrix:")
    print(display_df.to_string(index=True, header=True, justify='center', col_space=12))
    print('')
    return cm_df

def display_confusion_matrix(cm_df, risk_name, title_suffix="(merged: medium->risk)"):
    """
    打印给定cm_df的混淆矩阵（与load_confusion_matrix相同风格），用于合并后的结果。
    """
    display_df = cm_df.copy()
    display_df.index = [f"pred: {cls}" for cls in display_df.index]
    display_df.columns = [f"label: {cls}" for cls in display_df.columns]

    label_totals = cm_df.sum(axis=0)
    display_df.loc['label:total'] = label_totals.values
    pred_totals = display_df.sum(axis=1)
    display_df['pred:total'] = pred_totals.values

    display_df = display_df.astype(int)
    print(f"{' ':20} {risk_name} confusion matrix {title_suffix}:")
    print(display_df.to_string(index=True, header=True, justify='center', col_space=12))
    print('')

def calculate_f1_scores(cm_df):
    """计算每个类别的精确率、召回率和F1分数
    
    参数:
        cm_df: 混淆矩阵DataFrame，行是实际类别，列是预测类别
    
    返回:
        dict: 包含每个类别F1分数的字典
        dict: 包含每个类别精确率的字典
        dict: 包含每个类别召回率的字典
        float: 宏平均F1分数
    """
    # 获取类别列表
    classes = cm_df.index.tolist()
    
    # 初始化结果字典
    f1_scores = {}
    precision_scores = {}
    recall_scores = {}
    
    # 计算每个类别的指标
    for cls in classes:
        # 真正例(TP): 对角线元素
        tp = cm_df.loc[cls, cls]
        
        # 假正例(FP): 该列其他行的和
        fp = cm_df[cls].sum() - tp
        
        # 假负例(FN): 该行其他列的和
        fn = cm_df.loc[cls].sum() - tp
        
        # 计算精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores[cls] = f1
        precision_scores[cls] = precision
        recall_scores[cls] = recall
    
    # 计算宏平均F1分数
    macro_f1 = sum(f1_scores.values()) / len(f1_scores) if len(f1_scores) > 0 else 0
    
    return f1_scores, precision_scores, recall_scores, macro_f1

def _detect_columns_by_first_file(folder_path, file_patterns):
    """
    复用原有的“检测是否存在medium列”的逻辑 返回columns定义。
    """
    if file_patterns:
        first_pattern = file_patterns[0]
        first_file_path = os.path.join(folder_path, first_pattern)
        try:
            first_cm_df = pd.read_csv(first_file_path, index_col=0)
            classes = first_cm_df.index.tolist()
            if "medium" in classes:
                columns = ["no", "medium", "high", "overall"]
                print('find "medium"')
            else:
                columns = ["no", "high", "overall"]
                print('not find "medium"')
        except Exception as e:
            print(f"读取第一个文件 {first_file_path} 时出错: {e}")
            # 默认使用包含medium的列定义
            columns = ["no", "medium", "high", "overall"]
    else:
        # 如果没有文件，默认使用包含medium的列定义
        columns = ["no", "medium", "high", "overall"]
    print(f'set columns with {columns}')
    return columns

def merge_medium_high_cm(cm_df):
    """
    将cm_df中的 medium 和 high 合并为 risk（同时合并行与列）。
    若没有medium但有high，则将 high 重命名为 risk（用于二分类对齐）。
    """
    merged = cm_df.copy()
    has_medium = "medium" in merged.index and "medium" in merged.columns
    has_high = "high" in merged.index and "high" in merged.columns

    if has_medium and has_high:
        # 先合并列（预测维度）
        merged["risk"] = merged.get("medium", 0) + merged.get("high", 0)
        merged = merged.drop(columns=[c for c in ["medium", "high"] if c in merged.columns])
        # 再合并行（真实标签维度）
        merged.loc["risk"] = merged.loc["medium"] + merged.loc["high"]
        merged = merged.drop(index=[r for r in ["medium", "high"] if r in merged.index])
    elif (not has_medium) and has_high:
        # 二分类：直接把 high 重命名为 risk
        merged = merged.rename(index={"high": "risk"}, columns={"high": "risk"})
    # 若均不存在，则直接返回（可能是只有 False/No 的异常情况）
    return merged


def process_folder(folder_path):
    """处理文件夹中的所有混淆矩阵文件并生成结果DataFrame"""
    # 定义要查找的文件名模式
    file_patterns = [
        "confusion_matrix_for_attribute_abandonment.csv",
        "confusion_matrix_for_attribute_broken.csv",
        "confusion_matrix_for_attribute_corrosion.csv",
        "confusion_matrix_for_attribute_deformation.csv"
    ]
    
    # 提取risk名称的映射
    risk_names = {
        "confusion_matrix_for_attribute_abandonment.csv": "abandonment",
        "confusion_matrix_for_attribute_broken.csv": "broken",
        "confusion_matrix_for_attribute_corrosion.csv": "corrosion",
        "confusion_matrix_for_attribute_deformation.csv": "deformation"
    }
    
    columns = _detect_columns_by_first_file(folder_path, file_patterns)

    results_df = pd.DataFrame(index=risk_names.values(), columns=columns)
    results_df_precision = pd.DataFrame(index=risk_names.values(), columns=columns)
    results_df_recall = pd.DataFrame(index=risk_names.values(), columns=columns)
    
    # 处理每个文件
    for pattern in file_patterns:
        risk_name = risk_names[pattern]
        file_path = os.path.join(folder_path, pattern)
        
        try:
            # 加载混淆矩阵 + 打印（原有打印）
            cm_df = load_confusion_matrix(file_path, risk_name)
            
            # 计算指标
            f1_scores, precision_scores, recall_scores, macro_f1 = calculate_f1_scores(cm_df)
            
            # 填充结果DataFrame（原有映射：False -> 'no'）
            results_df.loc[risk_name, "no"] = f1_scores.get("False", 0)
            if "medium" in columns:
                results_df.loc[risk_name, "medium"] = f1_scores.get("medium", 0)
            results_df.loc[risk_name, "high"] = f1_scores.get("high", 0)
            results_df.loc[risk_name, "overall"] = macro_f1
            
            results_df_precision.loc[risk_name, "no"] = precision_scores.get("False", 0)
            if "medium" in columns:
                results_df_precision.loc[risk_name, "medium"] = precision_scores.get("medium", 0)
            results_df_precision.loc[risk_name, "high"] = precision_scores.get("high", 0)
            results_df_precision.loc[risk_name, "overall"] = sum(precision_scores.values()) / len(precision_scores) if precision_scores else 0
            
            results_df_recall.loc[risk_name, "no"] = recall_scores.get("False", 0)
            if "medium" in columns:
                results_df_recall.loc[risk_name, "medium"] = recall_scores.get("medium", 0)
            results_df_recall.loc[risk_name, "high"] = recall_scores.get("high", 0)
            results_df_recall.loc[risk_name, "overall"] = sum(recall_scores.values()) / len(recall_scores) if recall_scores else 0
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            # 填充NaN表示处理失败
            results_df.loc[risk_name] = [np.nan] * len(columns)
            results_df_precision.loc[risk_name] = [np.nan] * len(columns)
            results_df_recall.loc[risk_name] = [np.nan] * len(columns)
    
    return results_df, results_df_precision, results_df_recall



def process_folder_merged(folder_path):
    """
    在不改变原有结果的情况下，额外输出“medium+high -> risk”的
    二分类结果（no vs risk）。打印合并后的混淆矩阵，并返回合并版指标。
    """
    file_patterns = [
        "confusion_matrix_for_attribute_abandonment.csv",
        "confusion_matrix_for_attribute_broken.csv",
        "confusion_matrix_for_attribute_corrosion.csv",
        "confusion_matrix_for_attribute_deformation.csv"
    ]
    risk_names = {
        "confusion_matrix_for_attribute_abandonment.csv": "abandonment",
        "confusion_matrix_for_attribute_broken.csv": "broken",
        "confusion_matrix_for_attribute_corrosion.csv": "corrosion",
        "confusion_matrix_for_attribute_deformation.csv": "deformation"
    }

    # 合并版仅有两类：no（False）与 risk（medium+high）
    merged_cols = ["no", "risk", "overall"]
    merged_results = pd.DataFrame(index=risk_names.values(), columns=merged_cols)
    merged_precision = pd.DataFrame(index=risk_names.values(), columns=merged_cols)
    merged_recall = pd.DataFrame(index=risk_names.values(), columns=merged_cols)

    for pattern in file_patterns:
        risk_name = risk_names[pattern]
        file_path = os.path.join(folder_path, pattern)
        try:
            raw_cm = pd.read_csv(file_path, index_col=0)
            merged_cm = merge_medium_high_cm(raw_cm)

            # 打印合并后的混淆矩阵（独立标题）
            display_confusion_matrix(merged_cm, risk_name, title_suffix="(merged: medium+high → risk)")

            # 计算合并后的指标
            f1_scores, precision_scores, recall_scores, macro_f1 = calculate_f1_scores(merged_cm)

            merged_results.loc[risk_name, "no"] = f1_scores.get("False", 0)
            merged_results.loc[risk_name, "risk"] = f1_scores.get("risk", 0)
            merged_results.loc[risk_name, "overall"] = macro_f1

            merged_precision.loc[risk_name, "no"] = precision_scores.get("False", 0)
            merged_precision.loc[risk_name, "risk"] = precision_scores.get("risk", 0)
            merged_precision.loc[risk_name, "overall"] = sum(precision_scores.values()) / len(precision_scores) if precision_scores else 0

            merged_recall.loc[risk_name, "no"] = recall_scores.get("False", 0)
            merged_recall.loc[risk_name, "risk"] = recall_scores.get("risk", 0)
            merged_recall.loc[risk_name, "overall"] = sum(recall_scores.values()) / len(recall_scores) if recall_scores else 0

        except Exception as e:
            print(f"合并medium/high时出错（{file_path}）: {e}")
            merged_results.loc[risk_name] = [np.nan] * len(merged_cols)
            merged_precision.loc[risk_name] = [np.nan] * len(merged_cols)
            merged_recall.loc[risk_name] = [np.nan] * len(merged_cols)

    return merged_results, merged_precision, merged_recall

def risk_analysis(folder_path, output_f1='f1_results.csv', output_precision='precision_results.csv', output_recall='recall_results.csv'):

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"路径不是一个文件夹: {folder_path}")

    # 处理文件夹
    results_df, results_df_precision, results_df_recall = process_folder(folder_path)

    # 计算4个risk的overall指标平均值
    f1_overall_avg = results_df['overall'].mean()
    precision_overall_avg = results_df_precision['overall'].mean()
    recall_overall_avg = results_df_recall['overall'].mean()

    # 打印结果
    print(f"{' ':15}F1 result:")
    print(results_df)
    print(f"Average: {f1_overall_avg:.4f}\n")

    print(f"{' ':15}precision result:")
    print(results_df_precision)
    print(f"Average: {precision_overall_avg:.4f}\n")

    print(f"{' ':15}recall result:")
    print(results_df_recall)
    print(f"Average: {recall_overall_avg:.4f}\n")


    # 保存结果
    output_f1_path = os.path.join(folder_path, output_f1)
    results_df.to_csv(output_f1_path)
    print(f"\nF1结果已保存到: {output_f1_path}")

    output_precision_path = os.path.join(folder_path, output_precision)
    results_df_precision.to_csv(output_precision_path)
    print(f"精确率结果已保存到: {output_precision_path}")

    output_recall_path = os.path.join(folder_path, output_recall)
    results_df_recall.to_csv(output_recall_path)
    print(f"召回率结果已保存到: {output_recall_path}")


    if "medium" in results_df.columns:
        # ===== 额外新增：合并（medium+high -> risk）的二分类结果 =====
        merged_results, merged_precision, merged_recall = process_folder_merged(folder_path)

        # 打印合并后的指标表
        f1_overall_avg_m = merged_results['overall'].mean()
        precision_overall_avg_m = merged_precision['overall'].mean()
        recall_overall_avg_m = merged_recall['overall'].mean()

        print(f"\n{' ':10}F1 result (merged: medium+high → risk):")
        print(merged_results)
        print(f"Average: {f1_overall_avg_m:.4f}\n")

        print(f"{' ':10}precision result (merged):")
        print(merged_precision)
        print(f"Average: {precision_overall_avg_m:.4f}\n")

        print(f"{' ':10}recall result (merged):")
        print(merged_recall)
        print(f"Average: {recall_overall_avg_m:.4f}\n")

        # 保存合并后的结果到独立文件，避免覆盖原始结果
        merged_f1_path = os.path.join(folder_path, 'f1_results_merged.csv')
        merged_results.to_csv(merged_f1_path)
        print(f"合并F1结果已保存到: {merged_f1_path}")

        merged_precision_path = os.path.join(folder_path, 'precision_results_merged.csv')
        merged_precision.to_csv(merged_precision_path)
        print(f"合并精确率结果已保存到: {merged_precision_path}")

        merged_recall_path = os.path.join(folder_path, 'recall_results_merged.csv')
        merged_recall.to_csv(merged_recall_path)
        print(f"合并召回率结果已保存到: {merged_recall_path}")

if __name__ == '__main__':
    # main('/localnvme/project/ultralytics/runs/msegment/val117')
    main('/localnvme/project/ultralytics/runs/msegment/val193')
    # main('/localnvme/project/ultralytics/runs/msegment/val83')
    # main('/localnvme/project/ultralytics/runs/msegment/val84')