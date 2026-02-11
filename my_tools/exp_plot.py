import matplotlib.pyplot as plt
import numpy as np
import math
# --- 1. 设置学术风格 (可选) ---
# 尝试设置字体为 Times New Roman 或类似的衬线字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def calculate_std_dev(numbers):
    """
    计算一组数字的标准差
    """
    if not numbers:
        return 0

    n = len(numbers)

    # 计算平均值
    mean = sum(numbers) / n

    # 计算方差
    variance = sum((x - mean) ** 2 for x in numbers) / n

    # 计算标准差
    std_dev = math.sqrt(variance)

    print(f"数据: {numbers}")
    print(f"标准差: {std_dev:.4f}")
    return std_dev

# --- 2. 数据准备 ---
# 光照因子 (X轴)
illumination_factors = [0.5, 0.75, 1.0, 1.25, 1.5]

# 您的实验数据 (Y轴)
metrics = {
    'mAP50':    [0.7, 0.684, 0.698, 0.694, 0.663],
    'OA':       [0.97, 0.97, 0.971, 0.966, 0.966],
    'Macro-F1': [0.555, 0.561, 0.559, 0.544, 0.538],
    'Micro-F1': [0.645, 0.647, 0.651, 0.623, 0.619],
    'Precision': [0.61, 0.602, 0.611, 0.601, 0.547],
    'Recall': [0.548, 0.559, 0.553, 0.538, 0.537],
}

# 您计算出的标准差 (用于图例显示)
stds = {
    'mAP50': calculate_std_dev(metrics['mAP50']),
    'OA': calculate_std_dev(metrics['OA']),
    'Macro-F1': calculate_std_dev(metrics['Macro-F1']),
    'Micro-F1': calculate_std_dev(metrics['Micro-F1']),
    'Precision': calculate_std_dev(metrics['Precision']),
    'Recall': calculate_std_dev(metrics['Recall']),
}

# 定义颜色和标记，确保黑白打印也能区分
styles = {
    'OA':         {'color': '#2ca02c', 'marker': 'o', 'label': r'OA ($\sigma=0.0022$)'},          # 绿色，圆点
    'mAP50':      {'color': '#1f77b4', 'marker': 's', 'label': r'mAP50 ($\sigma=0.0136$)'},       # 蓝色，方块
    'Macro-F1':   {'color': '#ff7f0e', 'marker': '^', 'label': r'Macro-F1 ($\sigma=0.0089$)'},    # 橙色，三角
    'Micro-F1':   {'color': '#d62728', 'marker': 'D', 'label': r'Micro-F1 ($\sigma=0.0075$)'},    # 红色，菱形
    'Precision':  {'color': '#9467bd', 'marker': 'p', 'label': r'Precision ($\sigma=0.0102$)'},    # 紫色，五边形
    'Recall':     {'color': '#8c564b', 'marker': 'h', 'label': r'Recall ($\sigma=0.0114$)'}       # 棕色，六边形
}

# --- 3. 开始绘图 ---
plt.figure(figsize=(8, 6)) # 设置图片大小 (宽, 高)

# 绘制三条线
for name, data in metrics.items():
    plt.plot(illumination_factors, data,
             color=styles[name]['color'],
             marker=styles[name]['marker'],
             label=styles[name]['label'],
             linewidth=2,
             markersize=8,
             linestyle='-')

# --- 4. 装饰图表 ---
# 设置坐标轴标签 (支持LaTeX格式)
plt.xlabel(r'Illumination Factor ($\gamma$)')
plt.ylabel('Metric Score')

# 设置Y轴范围 (0.5到1.0能较好地展示数据，您可以根据需要调整)
plt.ylim(0.5, 1.05)

# 设置X轴刻度 (只显示测试过的点)
plt.xticks(illumination_factors)

# 添加网格 (透明度0.5)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加图例 (位置自动寻找最佳空白处)
plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.6), frameon=True)
# plt.legend(loc='best',frameon=True)
# 紧凑布局，防止标签被切掉
plt.tight_layout()

# --- 5. 保存与显示 ---
# 保存为高分辨率图片 (建议保存为 PDF 或 300dpi 的 PNG)
plt.savefig('robustness_evaluation.png', dpi=300, bbox_inches='tight')
plt.savefig('robustness_evaluation.pdf', bbox_inches='tight') # 矢量图适合投稿

plt.show()