import os
import cv2
import yaml
from tqdm import tqdm
import sys
parent_dir = os.path.dirname( os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import yaml_load
from pprint import pprint

def visualize_from_txt(
    video_path,
    labels_dir,
    output_video_path,
    data_path=None,
    with_conf=False,
    line_width=4,
):
    """
    读取视频 + YOLO txt（class xc yc w h id），
    使用 ultralytics 的 Annotator 进行可视化并保存为新视频。

    :param video_path: 原始视频路径
    :param labels_dir: 对应帧的 txt 目录（建议是重映射后的目录）
    :param output_video_path: 输出视频路径
    :param data_path: data，可选；不传则只显示 ID
    :param line_width: 画框线宽
    """
    if data_path is not None:
        data = yaml_load(data_path)
        class_names = data.get("names", None)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"video info: total frames={total_frames}, FPS={fps:.2f}, time={duration:.2f} second")
    
    if output_video_path.endswith(".avi"):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    elif output_video_path.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    frame_index = 1

    txt_files = os.listdir(labels_dir)
    txt_ids = [get_frame_index(txt_name) for txt_name in txt_files]
    txt_files_dict = dict(zip(txt_ids, txt_files))

    while True:
        print(f'processing frame {frame_index}/{total_frames}')
        ret, frame = cap.read()
        if not ret:
            break

        annotator = Annotator(frame, line_width=line_width)

        if frame_index in txt_files_dict:
            txt_path = os.path.join(labels_dir, txt_files_dict[frame_index])
            with open(txt_path, "r") as f:
                lines = f.read().strip().splitlines()

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()
                if with_conf:
                    cls, xc, yc, w_norm, h_norm, conf, tid = parts
                else:
                    cls, xc, yc, w_norm, h_norm, tid = parts
                    conf = None
                cls = int(cls)
                tid = int(tid)

                # 归一化坐标 -> 像素坐标
                xc = float(xc) * w
                yc = float(yc) * h
                ww = float(w_norm) * w
                hh = float(h_norm) * h

                x1 = int(xc - ww / 2)
                y1 = int(yc - hh / 2)
                x2 = int(xc + ww / 2)
                y2 = int(yc + hh / 2)

                box = [x1, y1, x2, y2]

                if class_names is None:
                    label = f"ID {tid}"
                else:
                    # 支持 list 或 dict
                    if isinstance(class_names, dict):
                        cname = class_names.get(cls, str(cls))
                    else:
                        cname = class_names[cls] if cls < len(class_names) else str(cls)
                    label = f"ID {tid}: {cname}"
                    if with_conf:
                        label += f" {float(conf):.2f}"

                annotator.box_label(box, label, color=colors(cls, True))

        frame_out = annotator.result()
        writer.write(frame_out)

        frame_index += 1

    cap.release()
    writer.release()
    print(f"可视化视频已保存到: {output_video_path}")


def get_frame_index(txt_name):
    frame_part = txt_name.rsplit('_', 1)[1]
    frame_id = int(frame_part.split('.')[0])
    return frame_id
    

def remap_track_ids_txt(src_dir, dst_dir, mannual_supply={}, with_conf=False):
    """
    从 src_dir 读取 YOLO+tracker 的 txt（class xc yc w h id），
    重新映射 id 为 1,2,3,... 连续编号，
    并将结果保存到 dst_dir（文件名保持不变）。

    :param src_dir: 原始 txt 目录（ultralytics 生成的 labels）
    :param dst_dir: 重映射后 txt 的输出目录
    :return: old_id -> new_id 的字典
    """
    os.makedirs(dst_dir, exist_ok=True)
    record_len = 7 if with_conf else 6

    id_map = {}
    next_new_id = 1

    txt_files = sorted(
        [f for f in os.listdir(src_dir) if f.endswith(".txt")],
        key=get_frame_index
    )

    for txt_name in tqdm(txt_files):
        src_path = os.path.join(src_dir, txt_name)
        dst_path = os.path.join(dst_dir, txt_name)

        with open(src_path, "r") as f:
            lines = f.read().strip().splitlines()

        new_lines = []
        for lid, line in enumerate(lines):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) < record_len:
                # 不符合 class xc yc w h (conf) id 格式就跳过
                print(f'\n{txt_name} len error: {parts}')
                if mannual_supply is not None and txt_name in mannual_supply and lid in mannual_supply[txt_name]:
                    parts.append(mannual_supply[txt_name][lid])
                    print(f'mannually supply as {parts[-1]}')
                else:
                    continue


            tid = int(parts[-1])

            if tid not in id_map:
                id_map[tid] = next_new_id
                next_new_id += 1
            new_tid = id_map[tid]
            parts[-1] = str(new_tid)

            new_line = ' '.join(parts)
            new_lines.append(new_line)

        with open(dst_path, "w") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")

    print(f"ID remap finish，total {len(id_map)} unique objects")
    pprint(id_map)
    return id_map


if __name__ == '__main__':
    pass
    video_path = r'/localnvme/data/bdd/DReality_data/video_data/V6_DJI_0344_W_CLIP1.MP4'
    data_path = r'/localnvme/project/ultralytics/ultralytics/cfg/bdd_dataset/dreality_1c_fv2_v3_rel.yaml'

    output_video_path = r'/localnvme/project/ultralytics/runs/detect/track9/V6_DJI_0344_W_CLIP1_remap.avi'
    input_label_dir = r'/localnvme/project/ultralytics/runs/detect/track9/labels'
    output_label_dir = r'/localnvme/project/ultralytics/runs/detect/track9/labels_remap'
    mannual_supply={
        'V6_DJI_0344_W_CLIP1_604.txt':{
            0: '16',
        },
        'V6_DJI_0344_W_CLIP1_999.txt':{
            0: '26',
        },
    }
    # mannual_supply = None
    # remap_track_ids_txt(input_label_dir, output_label_dir, mannual_supply=mannual_supply, with_conf=True)

    visualize_from_txt(video_path, output_label_dir, output_video_path, data_path=data_path, with_conf=True)