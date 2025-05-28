import demo_base
import torch

demo_base.TASK = 'classify'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = ".yaml"



if __name__ == '__main__':
    pass
