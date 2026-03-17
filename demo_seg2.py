import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 64
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata7961_seg_c5_l2_1022_re_80p_ref.yaml"
# demo_base.DATA = "testdata91_seg_c5_l2_1022_re_80p_ref.yaml"
demo_base.CONF_VAL = 0.001
demo_base.CONF_PREDICT = 0.1
if __name__ == '__main__':
    pass
    # demo_base.model_val(
    #     r'rip_seg-[yolov8x-seg]',
    # )
    demo_base.model_predict(
        r'rip_seg-[yolov8x-seg]',
        r'/data/huilin//scrinvme/huilin/bdd/cp_data/rip_seg/val_images',
        name=r'/data/huilin//scrinvme/huilin/bdd/cp_data/rip_seg/val_images_infer/a100_yolov8x-seg',
        save=False, save_conf=True,
    )
