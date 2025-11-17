import cv2
import numpy as np
import onnxruntime as ort

def get_model(input_path):
    model = ort.InferenceSession(input_path,
                                 providers=["CUDAExecutionProvider",
                                            "CPUExecutionProvider"])
    return model

def infer_img(session, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (960, 608))
    img = np.transpose(img, (2, 0, 1))[np.newaxis].astype(np.float32)/255.0
    img = np.repeat(img, 6, axis=0)
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    # print(len(outputs))
    print(outputs[1].shape)
    return outputs

if __name__ == '__main__':
    pass
    model_all = r'runs/msegment/fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3/weights/best.onnx'
    model_a = r'runs/msegment/fusedata7961_mseg_c5_l2_1111_v15_sa_test-[yolov10x-mseg-dlka3res-7-unet-single]/weights/best.onnx'
    model_b = r'runs/msegment/fusedata7961_mseg_c5_l2_1111_v15_sb_test-[yolov10x-mseg-dlka3res-7-unet-single]/weights/best.onnx'
    model_c = r'runs/msegment/fusedata7961_mseg_c5_l2_1111_v15_sc_test-[yolov10x-mseg-dlka3res-7-unet-single]/weights/best.onnx'
    model_d = r'runs/msegment/fusedata7961_mseg_c5_l2_1111_v15_sd_test-[yolov10x-mseg-dlka3res-7-unet-single]/weights/best.onnx'
    img_path = r'/scrinvme/huilin/isds/other_data/upload1014/Val_set/Val_set_upload/cam4/DA5324645_20250808112427399.jpg'
    # for model in [model_all, model_a, model_b, model_c, model_d]:
    #     session = get_model(model)
    #     infer_img(session, img_path)

    session1 = get_model(model_all)
    outputs1 = infer_img(session1, img_path)

    session2 = get_model(model_b)
    outputs2 = infer_img(session2, img_path)

    print()