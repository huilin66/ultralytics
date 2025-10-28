import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata7961_seg_c5_l2_1022_re_80p_ref.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo10('yolov10x-seg-dlka3res.yaml', auto_optim=False, use_fl_seg=True)

    # demo_base.model_val(
    #     r'fusedata7961_seg_c5_l2_1022_re_80p_ref-[yolov10x-seg-dlka3res]8',
    #     save_json=True
    # )
    demo_base.model_predict(
        r'fusedata7961_seg_c5_l2_1022_re_80p_ref-[yolov10x-seg-dlka3res]8',
        r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine/images',
        name=r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine/result_analysis/infer',
        conf=0.3,
    )
