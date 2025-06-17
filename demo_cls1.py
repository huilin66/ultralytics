import demo_base
import torch

demo_base.TASK = 'classify'
demo_base.EPOCHS = 50
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 32
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = ".yaml"



if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_riskA.yaml",)
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_riskB.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_riskC.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_riskD.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_update_riskA.yaml",)
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_update_riskB.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_update_riskC.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_update_riskD.yaml")
    
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_abandonment_data626box_cls.yaml", )
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_broken_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_corrosion_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_deformation_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_abandonment_data626box_cls.yaml", )
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_broken_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_corrosion_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_deformation_data626box_cls.yaml")