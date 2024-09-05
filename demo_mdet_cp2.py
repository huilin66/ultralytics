import demo_mdet
demo_mdet.EPOCHS = 3
if __name__ == '__main__':
    demo_mdet.model_val(weight_path=r'runs/mdetect/train291/weights/best.pt')
    demo_mdet.model_val(weight_path=r'runs/mdetect/train292/weights/best.pt')
    # demo_mdet.model_val(weight_path=r'runs/mdetect/train155/weights/best.pt')
    # demo_mdet.model_val(weight_path=r'runs/mdetect/train188/weights/best.pt')
    # demo_mdet.model_val(weight_path=r'runs/mdetect/train273/weights/best.pt')
    # demo_mdet.model_val(weight_path=r'runs/mdetect/train277/weights/best.pt')
    # demo_mdet.model_predict(
    #     weight_path=r'runs/mdetect/train151/weights/best.pt',
    #     img_dir=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/images'
    # )
    # demo_mdet.model_predict(
    #     weight_path=r'runs/mdetect/train152/weights/best.pt',
    #     img_dir=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/images'
    # )
    # demo_mdet.model_predict(
    #     weight_path=r'runs/mdetect/train155/weights/best.pt',
    #     img_dir=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/images'
    # )
    # demo_mdet.model_predict(
    #     weight_path=r'runs/mdetect/train286/weights/best.pt',
    #     img_dir=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/images'
    # )
