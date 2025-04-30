from demo_mseg import *

if __name__ == '__main__':
    pass
    # yolo8x('yolov8x-mseg.yaml', auto_optim=False, name=f'billboard_mseg_307')
    # yolo8x('yolov8x-mseg-p2.yaml', auto_optim=False, name=f'debug')
    # yolo8x('yolov8x-mseg-p6.yaml', auto_optim=False, name=f'billboard_mseg_307')
    # model_val('/nfsv4/23039356r/repository/ultralytics/runs/msegment/billboard_mseg_307/weights/best.pt')
    # model_val('/nfsv4/23039356r/repository/ultralytics/runs/msegment/billboard_mseg_3072/weights/best.pt')
    # model_val('/nfsv4/23039356r/repository/ultralytics/runs/msegment/billboard_mseg_3073/weights/best.pt')

    yolo8x('yolov8x-mseg.yaml', auto_optim=False, name=f'billboard_mseg_307', retrain=True,
           weight_path=r'/nfsv4/23039356r/repository/ultralytics/runs/msegment/billboard_mseg_3072/weights/best.pt')