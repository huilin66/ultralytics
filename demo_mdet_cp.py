import demo_mdet
import torch
demo_mdet.DEVICE = torch.device('cuda:1')
if __name__ == '__main__':
    demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf1.yaml')
    demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf1res.yaml')
    demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf2.yaml')
    demo_mdet.myolo10x(r'yolov10x-mdetect-sppftf2res.yaml')