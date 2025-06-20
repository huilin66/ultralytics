import demo_base
import torch

demo_base.TASK = 'classify'
demo_base.EPOCHS = 50
demo_base.IMGSZ = 640
demo_base.BATCH_SIZE = 64
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = ".yaml"



if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_riskA.yaml",)
    demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_riskB.yaml")
    demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_riskC.yaml")
    demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_riskD.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_update_riskA.yaml",)
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_update_riskB.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_update_riskC.yaml")
    # demo_base.yolo8('yolov8x-cls.yaml', weight_path='yolov8x-cls.pt', auto_optim=False, data="fusedata1361_cls_c6_check0618_update_riskD.yaml")
    
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_abandonment_data626box_cls.yaml", )
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_broken_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_corrosion_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', data="risk_deformation_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_abandonment_data626box_cls.yaml", )
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_broken_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_corrosion_data626box_cls.yaml")
    # demo_base.yolo11('yolo11n-cls.yaml', weight_path='yolo11n-cls.pt', auto_optim=False, data="risk_deformation_data626box_cls.yaml")

    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskA-[yolov8x-cls]4/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskA.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskA-[yolov8x-cls]4/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskA_pred_val.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskA-[yolov8x-cls]4/weights/best.pt',
    #                     data='fusedata1361_cls_c6_update_riskA.yaml')
    #
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskB-[yolov8x-cls]2/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskB.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskB-[yolov8x-cls]2/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskB_pred_val.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskB-[yolov8x-cls]2/weights/best.pt',
    #                     data='fusedata1361_cls_c6_update_riskB.yaml')
    #
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskC-[yolov8x-cls]2/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskC.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskC-[yolov8x-cls]2/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskC_pred_val.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskC-[yolov8x-cls]2/weights/best.pt',
    #                     data='fusedata1361_cls_c6_update_riskC.yaml')
    #
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskD-[yolov8x-cls]/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskD.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskD-[yolov8x-cls]/weights/best.pt',
    #                     data='fusedata1361_cls_c6_riskD_pred_val.yaml')
    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_riskD-[yolov8x-cls]/weights/best.pt',
    #                     data='fusedata1361_cls_c6_update_riskD.yaml')

    # demo_base.model_val(r'runs/classify/fusedata1361_cls_c6_update_riskD-[yolov8x-cls]/weights/best.pt')