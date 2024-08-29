import demo_mdet
import torch
demo_mdet.DEVICE = torch.device('cuda:1')
if __name__ == '__main__':
    # demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf1.yaml', auto_optim=False)
    # demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf1res.yaml', auto_optim=False)
    # demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf2.yaml', auto_optim=False)
    # demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf2res.yaml', auto_optim=False)
    # demo_mdet.myolo10x(r'yolov10x-mdetect-elantf2.yaml', auto_optim=False)
    # demo_mdet.myolo10x(r'yolov10x-mdetect-elantf2res.yaml', auto_optim=False)

    demo_mdet.myolo10x(r'yolov10x-mdetect-psatf1.yaml', auto_optim=False)
    demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf1res.yaml', auto_optim=False)
    demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf2.yaml', auto_optim=False)
    demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf2res.yaml', auto_optim=False)