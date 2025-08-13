from demo4paper import *


if __name__ == '__main__':
    pass
    # import cv2
    # from ultralytics import solutions
    # heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model='runs/exp_results/exp_mayolox_/weights/best.pt')
    # image = cv2.imread('/localnvme/data/billboard/data0806_m/paper_demo/images/FLIR0904.png')
    # result = heatmap(image)
    # result_img = result.plot_im
    # cv2.imwrite('demo_heatmap.png', image)
    # print(result_img.shape)
    # print(result)
    model = YOLO('runs/exp_results/exp_yolo10x/weights/best.pt', task=TASK)
    model.predict(
        '/localnvme/data/billboard/data0806_m/paper_demo/images/FLIR0904.png',
        save=True,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
        save_txt=True,
        save_conf=True,
        visualize=True
    )