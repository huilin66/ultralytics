import os
import pandas as pd
from matplotlib import pyplot as plt


def get_df(csv_path):
    df = pd.read_csv(csv_path, header=0, index_col=0)
    return df

def metric_plot(df, col_name='metrics/mAP50(B)', topk=5):
    ax = df[col_name].plot(kind='line', figsize=(12, 6))

    if topk:
        topk_max = df[col_name].nlargest(topk)
        print(topk_max)
        ax.scatter(topk_max.index, topk_max.values, color='red', s=50, zorder=5)
        for idx, val in topk_max.items():
            ax.vlines(x=idx, ymin=df[col_name].min(), ymax=val, color='black', linestyles='--', alpha=0.3, linewidth=0.8)
            ax.hlines(y=val, xmin=df.index.min(), xmax=idx, color='black', linestyles='--', alpha=0.3, linewidth=0.8)
            ax.annotate(f'{val:.3f}', (idx, val), xytext=(10, 10), textcoords='offset points', fontsize=8, color='red')
    plt.show()

if __name__ == '__main__':
    pass
    train_name = 'fusedata7961_seg_c5_1106_v12_src-[yolov10x-seg-dlka3res]'
    task = 'segment'
    train_path = os.path.join('../runs', task, train_name, 'results.csv')
    df = get_df(train_path)
    metric_plot(df)