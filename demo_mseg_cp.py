import demo_mseg
import torch
demo_mseg.DEVICE = torch.device('cuda:0')
demo_mseg.DATA = "billboard_mseg_389_c6.yaml"
demo_mseg.EPOCHS = 100
if __name__ == '__main__':
    pass
    # demo_mseg.yolo8x('yolov8x-mseg-8.yaml', auto_optim=False, name=f'billboard_mseg_389', retrain=True,
    #                  data='billboard_mseg_389_filter001.yaml',
    #                  weight_path=r'runs/segment/billboard_seg_3895/weights/best.pt')
    # demo_mseg.yolo8x('yolov8x-mseg-8.yaml', auto_optim=False, name=f'billboard_mseg_389', retrain=True,
    #                  data='billboard_mseg_389_filter005.yaml',
    #                  weight_path=r'runs/segment/billboard_seg_3895/weights/best.pt')
    # demo_mseg.yolo8x('yolov8x-mseg-8.yaml', auto_optim=False, name=f'billboard_mseg_389', retrain=True,
    #                  data='billboard_mseg_389_filter010.yaml',
    #                  weight_path=r'runs/segment/billboard_seg_3895/weights/best.pt')
    # demo_mseg.yolo8x('yolov8x-mseg-7.yaml', auto_optim=False, name=f'billboard_mseg_389', retrain=True,
    #                  data='billboard_mseg_389_filter001_c6.yaml',
    #                  weight_path=r'runs/segment/billboard_seg_3896/weights/best.pt')
    # demo_mseg.yolo8x('yolov8x-mseg-7.yaml', auto_optim=False, name=f'billboard_mseg_389', retrain=True,
    #                  data='billboard_mseg_389_filter005_c6.yaml',
    #                  weight_path=r'runs/segment/billboard_seg_3897/weights/best.pt')
    demo_mseg.yolo8x('yolov8x-mseg-7.yaml', auto_optim=False, name=f'billboard_mseg_389', retrain=True,
                     data='billboard_mseg_389_filter010_c6.yaml',
                     weight_path=r'runs/segment/billboard_meg_389/weights/best.pt')

    # demo_mseg.model_val(r'runs/msegment/billboard_mseg_38919/weights/best.pt',
    #                     data='billboard_mseg_389_filter001_c6.yaml')
    # demo_mseg.model_val(r'runs/msegment/billboard_mseg_38920/weights/best.pt',
    #                     data='billboard_mseg_389_filter005_c6.yaml')
    # demo_mseg.model_val(r'runs/msegment/billboard_mseg_38921/weights/best.pt',
    #                     data='billboard_mseg_389_filter010_c6.yaml')
    # demo_mseg.model_val(r'runs/msegment/billboard_mseg_38922/weights/best.pt',
    #                     data='billboard_mseg_389_filter001_c6.yaml')
    # demo_mseg.model_val(r'runs/msegment/billboard_mseg_38923/weights/best.pt',
    #                     data='billboard_mseg_389_filter005_c6.yaml')
    # demo_mseg.model_val(r'runs/msegment/billboard_mseg_38924/weights/best.pt',
    #                     data='billboard_mseg_389_filter010_c6.yaml')