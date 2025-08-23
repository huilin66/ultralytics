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
    
    # 确定列定义
    # 先检查第一个文件的类别结构
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

    # 创建结果DataFrame
    results_df = pd.DataFrame(index=risk_names.values(), columns=columns)
    results_df_precision = pd.DataFrame(index=risk_names.values(), columns=columns)
    results_df_recall = pd.DataFrame(index=risk_names.values(), columns=columns)
    
    # 处理每个文件
    for pattern in file_patterns:
        # 提取risk名称
        risk_name = risk_names[pattern]
        
        # 构建完整文件路径
        file_path = os.path.join(folder_path, pattern)
        
        try:
            # 加载混淆矩阵
            cm_df = load_confusion_matrix(file_path, risk_name)
            
            # 计算指标
            f1_scores, precision_scores, recall_scores, macro_f1 = calculate_f1_scores(cm_df)
            
            # 填充结果DataFrame
            # 注意: 这里假设类别顺序是False, medium, high
            # 我们将False映射到'no'
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


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理文件夹中的混淆矩阵文件并计算评估指标')
    parser.add_argument('--folder_path', type=str, default='/localnvme/project/ultralytics/runs/msegment/val287', help='包含混淆矩阵CSV文件的文件夹路径')
    parser.add_argument('--output_f1', type=str, default='f1_results.csv', help='F1结果输出文件名(默认: f1_results.csv)')
    parser.add_argument('--output_precision', type=str, default='precision_results.csv', help='精确率结果输出文件名(默认: precision_results.csv)')
    parser.add_argument('--output_recall', type=str, default='recall_results.csv', help='召回率结果输出文件名(默认: recall_results.csv)')
    args = parser.parse_args()
    
    try:
        # 检查文件夹是否存在
        if not os.path.isdir(args.folder_path):
            raise NotADirectoryError(f"路径不是一个文件夹: {args.folder_path}")
        
        # 处理文件夹
        results_df, results_df_precision, results_df_recall = process_folder(args.folder_path)
        
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
        output_f1_path = os.path.join(args.folder_path, args.output_f1)
        results_df.to_csv(output_f1_path)
        print(f"\nF1结果已保存到: {output_f1_path}")
        
        output_precision_path = os.path.join(args.folder_path, args.output_precision)
        results_df_precision.to_csv(output_precision_path)
        print(f"精确率结果已保存到: {output_precision_path}")
        
        output_recall_path = os.path.join(args.folder_path, args.output_recall)
        results_df_recall.to_csv(output_recall_path)
        print(f"召回率结果已保存到: {output_recall_path}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()