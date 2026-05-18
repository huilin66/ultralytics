import os
from yolo2mask import txt_to_mask
from mask_compare import generate_comparison_vis
from eval import result_evaluate
DATASET_NAME = "mars-seg.yaml"
CONF_VAL = 0.5
IMAGES_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/images'
MASKS_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/val/masks'
LABELS_TXT_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results/labels'
OUTPUT_MASK_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results/masks'
VAL_RESULT = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results'

def main():
    MODEL_LIST = [
        "yolo26x-seg.yaml",
        "yolov8x-seg.yaml",
        "yolov9e-seg.yaml",
        "yolov10x-seg.yaml",
        "yolo11x-seg.yaml",
        "yolo12x-seg.yaml",
        ]
    for MODEL_NAME in MODEL_LIST:
        NAME = f'{DATASET_NAME.split(".")[0]}_{MODEL_NAME.split(".")[0]}2'
        print(f'{"=="*20} [ANALYZE] {NAME} {"=="*20}')
        VAL_RESULT_DIR  = os.path.join(VAL_RESULT, f'{NAME}_{CONF_VAL}')
        PRED_TXT_DIR = os.path.join(VAL_RESULT_DIR, 'labels')
        PRED_MASK_DIR = os.path.join(VAL_RESULT_DIR, 'masks')
        VAL_RESULT_COMPARE_DIR = os.path.join(VAL_RESULT_DIR, 'masks_compare')
        txt_to_mask(IMAGES_DIR, PRED_TXT_DIR, PRED_MASK_DIR)
        generate_comparison_vis(IMAGES_DIR, MASKS_DIR, PRED_MASK_DIR, VAL_RESULT_COMPARE_DIR)
        result_evaluate(MASKS_DIR, PRED_MASK_DIR)
        print(f'{"=="*60}\n')

if __name__ == '__main__':
    main()
