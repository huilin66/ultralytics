import torch
import numpy as np
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:0")

def load_demo(path1, path2=None):
    data1 = torch.load(path1)
    if path2 is not None:
        data2 = torch.load(path2)
    print()

def compare_mseg_seg_ckpt(path1, path2, equal_check=True):
    model_a_state_dict = torch.load(path1)
    model_b_state_dict = torch.load(path2)
    # 初始化存储不一致的层
    inconsistent_layers = []

    # 比较两个模型的参数
    if equal_check:
        for (name_a, param_a), (name_b, param_b) in zip(model_a_state_dict.items(), model_b_state_dict.items()):
            if name_a != name_b:
                print(f"{name_a} vs {name_b}")
            else:
                # 检查参数是否一致
                if not torch.equal(param_a, param_b):
                    inconsistent_layers.append([name_a, param_a.shape, param_b.shape])
    else:
        for name_a, param_a in model_a_state_dict.items():
            if name_a in model_b_state_dict:
                param_b = model_b_state_dict[name_a]
                if torch.equal(param_a, param_b):
                    continue
                else:
                    inconsistent_layers.append([name_a, param_a.shape, param_b.shape])
            else:
                inconsistent_layers.append([name_a, param_a.shape, None])
    # 输出不一致的层
    if inconsistent_layers:
        print("Inconsistent layers found:")
        for layer in inconsistent_layers:
            print(layer)
        print(f'total length: {len(inconsistent_layers)} in {len(model_a_state_dict)}')
    else:
        print("All layers are consistent.")

def compare_mseg_seg_data(path1, path2):
    data_a = torch.load(path1).to(device1)
    data_b = torch.load(path2).to(device2)

    result = torch.equal(data_a, data_b)
    print(result)

def compare_mseg_seg_pred1(path1, path2):
    pred_a = torch.load(path1)
    pred_b = torch.load(path2)
    print('+++++++++++++++F0+++++++++++++++++')
    f_a = pred_a[0]
    f_b = torch.cat([
        pred_b[0][:, :11, :],
        pred_b[0][:, 15:, :],
        ], dim=1)
    result = torch.equal(f_a, f_b)
    print(result)
    result0 = torch.all(f_a==f_b, dim=(1, 2))
    result1 = torch.all(f_a==f_b, dim=(0, 2))
    print(result0)
    print(result1)


    print('+++++++++++++++F1-0-0+++++++++++++++++')
    f_a = pred_a[1][0][0]
    f_b = pred_b[1][0][0][:, :-4]
    result = torch.equal(f_a, f_b)
    print(result)
    print('+++++++++++++++F1-0-1+++++++++++++++++')
    f_a = pred_a[1][0][1]
    f_b = pred_b[1][0][1][:, :-4]
    result = torch.equal(f_a, f_b)
    print(result)
    print('+++++++++++++++F1-0-2+++++++++++++++++')
    f_a = pred_a[1][0][2]
    f_b = pred_b[1][0][2][:, :-4]
    result = torch.equal(f_a, f_b)
    print(result)
    print('+++++++++++++++F1-1+++++++++++++++++')
    f_a = pred_a[1][1]
    f_b = pred_b[1][1]
    result = torch.equal(f_a, f_b)
    print(result)

    print('+++++++++++++++F1-2+++++++++++++++++')
    f_a = pred_a[1][2]
    f_b = pred_b[1][2]
    result = torch.equal(f_a, f_b)
    print(result)

def compare_mseg_seg_pred2(path1, path2):
    pred_a = torch.load(path1)
    pred_b = torch.load(path2)
    print('+++++++++++++++F0+++++++++++++++++')
    for f_a, f_b in zip(pred_a[0], pred_b[0]):
        f_b = torch.cat([
            f_b[:, :6],
            f_b[:, 10:],
        ], dim=1)
        result = torch.equal(f_a, f_b)
        print(result)
        # result0 = torch.all(f_a == f_b, dim=(1))
        # result1 = torch.all(f_a == f_b, dim=(0))
        # print(result0)
        # print(result1)

    pred_b = torch.load(path2)
    print('+++++++++++++++F1+++++++++++++++++')
    f_a = pred_a[1]
    f_b = pred_b[1]
    result = torch.equal(f_a, f_b)
    print(result)

def compare_mseg_seg_loss():
    loss_a = torch.load('loss_seg2000.pth')
    loss_b = torch.load('loss_mseg2000.pth')

    result = torch.equal(loss_a[0], loss_b[0])
    print(result)
    print(loss_a)
    print(loss_b)
    result = torch.equal(loss_a[1], loss_b[1][:4])
    print(result)

def compare_mseg_seg_feat(path_a, path_b):
    f_a = torch.load(path_a)
    f_b = torch.load(path_b)
    result = torch.equal(f_a, f_b)
    print(path_a, path_b, result)

def compare_mseg_seg_feat1(path_a, path_b):
    f_a = torch.load(path_a)
    f_b = torch.load(path_b)
    f_b = torch.cat([
        f_b[:, :6],
        f_b[:, 10:],
    ], dim=1)
    result = torch.equal(f_a, f_b)
    print(path_a, path_b, result)

    result0 = torch.all(f_a == f_b, dim=(1))
    result1 = torch.all(f_a == f_b, dim=(0))
    print(result0)
    print(result1)

def compare_mseg_seg_preds(path_a, path_b):
    pred_a = torch.load(path_a, map_location=device1)['one2many']
    pred_b = torch.load(path_b, map_location=device2)

    print('+++++++++++++++F0+++++++++++++++++')
    f_a = pred_a[0]
    f_b = pred_b[0]

    result = torch.equal(f_a, f_b)
    print(result)


def compare_data(pt_path, npy_path):
    pass
    pt_data = torch.load(pt_path)
    npy_data = np.load(npy_path)

    print()

if __name__ == '__main__':
    pass
    # compare_mseg_seg_ckpt('seg/model.pt', 'mseg/model.pt', False)
    # compare_mseg_seg_data('seg/img.pt', 'mseg/img.pt')
    # compare_mseg_seg_pred1('seg/preds1.pt', 'mseg/preds1.pt')
    # compare_mseg_seg_pred2('seg/preds2.pstatt', 'mseg/preds2.pt')
    # compare_mseg_seg_pred2('seg/preds2.pt', 'mseg/preds2.pt')
    # compare_mseg_seg_feat('seg/bbox.pt', 'mseg/bbox.pt')
    # compare_mseg_seg_feat('seg/cls.pt', 'mseg/cls.pt')
    # compare_mseg_seg_feat1('seg/predn.pt', 'mseg/predn.pt')

    # compare_mseg_seg_data('seg1/img.pt', 'seg2/img.pt')
    # compare_mseg_seg_preds('seg1/preds.pt', 'seg2/preds.pt')

    # load_demo('seg1/input_x.pt', 'seg2/input_x.pt')

    # compare_data(r'/my_tools/data_onnx/20250812150605700.pt',
    #              r'/my_tools/data_onnx/20250812150605700.npy')


    # data11 = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_src.npy')
    # data12 = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_rs.npy')
    data1_letterbox = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_letterbox.npy')
    data1_input_int = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_input_int.npy')
    data1_input_float = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_input_float.npy')
    data1_input = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_input.npy')[0:1]
    data1_output0 = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_output0.npy')[0:1]
    data1_output1 = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_output1.npy')
    data1_output2 = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_output2.npy')
    data1_output3 = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_output3.npy')
    data1_output4 = np.load(r'/localnvme/project/ultralytics/my_tools/data_onnx/20250812150605700_output4.npy')

    # data21 = np.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_src.npy')
    # data22 = np.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_rs.npy')
    # data221 = np.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_rs1.npy')
    data2_letterbox = np.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_letterbox.npy')
    # data232 = torch.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_letterbox.pt')
    data2_input_int = torch.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_input_int.pt').permute(0, 2,3,1)[0].cpu().numpy()
    data2_input_float = torch.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_input_float.pt').cpu().numpy()
    data2_input = torch.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_input.pt')
    data2_output = torch.load(r'/localnvme/project/ultralytics/my_tools/data_torch/20250812150605700_output.pt')
    data2_output0 = data2_output[0].cpu().numpy()


    # result1 = (data11==data21).all()
    # print(result1)
    # result2 = (data12==data22).all()
    # print(result2)
    # result21 = (data12==data221).all()
    # print(result21)
    result_letterbox = (data1_letterbox==data2_letterbox).all()
    print(result_letterbox)
    result_input_int = (data1_input_int==data2_input_int).all()
    print(result_input_int)
    result_input_float = np.allclose(data1_input_float,data2_input_float, atol=1e-10)
    print(result_input_float)

    max_diff = (data1_output0-data2_output0).max()
    mean_diff = (data1_output0-data2_output0).mean()
    result_output_0 = np.allclose(data1_output0, data2_output0, atol=1e-3)
    print(result_output_0)
    print()
    # result3 = (data13==data23).all()
    # print(result3)
    # data32_tp = data232.permute(0, 2,3,1)[0].cpu().numpy()
    # result32 = np.allclose(data13, data32_tp, atol=1e-6)
    # print(result32)
    # data32_tp = data233.permute(0, 2,3,1)[0].cpu().numpy()
    # result32 = np.allclose(data13, data32_tp, atol=1e-6)
    # print(result32)
    # result242 = (data24==data242).all()
    # print(result242)
    # result4 = np.allclose(data14, data24.cpu().numpy(), atol=1e-6)
    # print(result4)
    # result50 = np.allclose(data150, data250, atol=1e-1)
    # print(result50)