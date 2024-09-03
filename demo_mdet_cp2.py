import demo_mdet
demo_mdet.EPOCHS = 3
if __name__ == '__main__':
    demo_mdet.model_val(weight_path=r'runs/mdetect/train151/weights/best.pt')
    demo_mdet.model_val(weight_path=r'runs/mdetect/train152/weights/best.pt')
    demo_mdet.model_val(weight_path=r'runs/mdetect/train155/weights/best.pt')
    demo_mdet.model_val(weight_path=r'runs/mdetect/train188/weights/best.pt')
    demo_mdet.model_val(weight_path=r'runs/mdetect/train273/weights/best.pt')
    demo_mdet.model_val(weight_path=r'runs/mdetect/train277/weights/best.pt')