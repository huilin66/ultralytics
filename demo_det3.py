import demo_base
import torch

demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:0')
demo_base.BATCH_SIZE = 8
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5

if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="dreality_1c_fv2_v3_rel.yaml", name='debug', epochs=1, batch=4)
    # img_dir = r'/localnvme/data/bdd/DReality_data/yolo_clip_v2/images'
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="dreality_1c_fv1.yaml")
    # demo_base.model_val('dreality_1c_fv2_v3-[yolov8x]')
    # demo_base.model_val('dreality_1c_fv1-[yolov8x]7')
    # demo_base.model_predict('dreality_1c_fv1-[yolov8x]6',
    #                         img_dir,
    #                         name=img_dir+'_infer',
    #                         batch=16, save_txt=True, conf=0.01,
    #                         )
    # demo_base.model_predict('dreality_1c_fv1-[yolov8x]7',
    #                         img_dir,
    #                         name=img_dir+'_infer',
    #                         batch=16, save_txt=True, conf=0.01,
    #                         )
    # demo_base.model_predict('dreality_1c_fv1-[yolov8x]7',
    #                         img_dir,
    #                         name=img_dir+'_infer',
    #                         batch=16, save_txt=True, conf=0.01, imgsz=1280,
    #                         )
    # demo_base.model_predict('dreality_1c_fv2-[yolov8x]',
    #                         img_dir,
    #                         name=img_dir+'_infer',
    #                         batch=16, save_txt=True, conf=0.01,
    #                         )
    # demo_base.model_predict('dreality_1c_fv2-[yolov8x]2',
    #                         img_dir,
    #                         name=img_dir+'_infer',
    #                         batch=16, save_txt=True, conf=0.01,
    #                         )
    # demo_base.model_predict('dreality_1c_fv2-[yolov8x]2',
    #                         img_dir,
    #                         name=img_dir+'_infer',
    #                         batch=16, save_txt=True, conf=0.01, imgsz=1280,
    #                         )
    demo_base.model_track('dreality_1c_fv2_v3-[yolov8x]',
                            r'/localnvme/data/bdd/DReality_data/video_data/V6_DJI_0344_W_CLIP1.MP4',
                            # tracker='bytetrack.yaml',
                            tracker='botsort.yaml',
                            imgsz=1280,
                            batch=32, save_txt=True, conf=0.3, save_conf=True
                            )