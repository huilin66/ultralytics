import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata1361_mseg_c6.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #                 data="psdata735_mseg_c6.yaml",
    #        weight_path=r'runs/segment/psdata735_seg_c6-[yolov8x-seg]3/weights/best.pt', )
    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #                 data="psdata735_mseg_c6_update.yaml",
    #        weight_path=r'runs/segment/psdata735_seg_c6-[yolov8x-seg]3/weights/best.pt', )
    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #         data="fusedata1361_mseg_c6.yaml",
    #        weight_path=r'runs/segment/fusedata1361_seg_c6-[yolov8x-seg]3/weights/best.pt')
    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #         data="fusedata1361_mseg_c6_update.yaml",
    #        weight_path=r'runs/segment/fusedata1361_seg_c6-[yolov8x-seg]3/weights/best.pt')



    demo_base.model_val(r'runs/msegment/fusedata1361_mseg_c6_update-[yolov8x-mseg-7]/weights/best.pt',
                        data='fusedata1361_mseg_c6_update.yaml',
                        save_txt=True, save_npy=True
                        )



    # demo_base.model_val(r'runs/msegment/fusedata870_mseg_c6-[yolov8x-mseg-7]2/weights/best.pt', data='fusedata870_mseg_c6.yaml')
    # demo_base.model_val(r'runs/msegment/fusedata870_mseg_c6-[yolov8x-mseg-7]2/weights/best.pt', data='fusedata870_mseg_c6_f010_ref.yaml')
    # demo_base.yolo8x('yolov8x-mseg-dlka3res-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=False,
    #        weight_path=r'runs/segment/billboard_seg_389_c618/weights/best.pt', )
    # demo_base.yolo8x('yolov8x-mseg-dlka3res-7.yaml', auto_optim=False, retrain=True,  mloss_mask=True, mloss_weight=True,
    #        weight_path=r'runs/segment/billboard_seg_389_c618/weights/best.pt',)
    # demo_base.model_predict('runs/msegment/billboard_mseg_389re1_c6-yolov8x-mseg-7/weights/best.pt',
    #                         '/nfsv4/23039356r/data/billboard/bd_data/data389re1_mseg_c6/images')


    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]2/weights/best.pt',
    #                     data='billboard_mseg_389_c6.yaml')
    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]3/weights/best.pt',
    #                     data='billboard_mseg_389_c6.yaml')
    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                     data='billboard_mseg_389_c6.yaml')
    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                     data='billboard_mseg_389_c6.yaml', conf=0.45)
    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                     data='billboard_mseg_389_c6.yaml', conf=0.4)
    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                     data='billboard_mseg_389_c6.yaml', conf=0.35)
    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                     data='billboard_mseg_389_c6.yaml', conf=0.3)
    # demo_base.model_predict(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                         r'/localnvme/data/billboard/ps_data/0516/images_split/left',
    #                         name='/localnvme/data/billboard/ps_data/0516/images_split_pred/left',
    #                         data='billboard_mseg_389_c6.yaml')
    # demo_base.model_predict(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                         r'/localnvme/data/billboard/ps_data/0516/images_split/right',
    #                         name='/localnvme/data/billboard/ps_data/0516/images_split_pred/right',
    #                         data='billboard_mseg_389_c6.yaml')
    #
    # demo_base.model_predict(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                         r'/localnvme/data/billboard/ps_data/0516/images',
    #                         name='/localnvme/data/billboard/ps_data/0516/images_pred',
    #                         data='billboard_mseg_389_c6.yaml')

    # demo_base.model_export(r'runs/msegment/fusedata870_mseg_c6-[yolov8x-mseg-7]2/weights/best.pt', imgsz=(608,960))
    # demo_base.model_predict(r'runs/msegment/fusedata870_mseg_c6-[yolov8x-mseg-7]2/weights/best.pt',
    #                         r'/localnvme/data/billboard/ps_data/test_20250527021438500.jpg')


    # demo_base.model_val(r'runs/msegment/billboard_mseg_389_c6-[yolov8x-mseg-dlka3res-7]4/weights/best.pt', data='billboard_mseg_389_c6.yaml')
    # demo_base.model_val(r'runs/msegment/fusedata1037_mseg_c6-[yolov8x-mseg-7]2/weights/best.pt')