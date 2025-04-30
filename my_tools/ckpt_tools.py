import torch


# save_path = '/nfsv4/23039356r/repository/ultralytics/6.pth'
# torch.save(self.model.state_dict(), save_path)

def remove_c4():
    w1_path = r'../runs/mdetect/exp_yolo10x_m34/weights/best.pt'
    ckpt = torch.load(w1_path)
    # model = ckpt['model']
    # for m in model.model.modules():
    #     print(m)
    state_dict = ckpt['model'].state_dict()
    layer_name = 'model.23.cv4'  # 根据具体模型的层次结构，layer23 可能需要调整名称
    keys_to_remove = [key for key in state_dict if key.startswith(layer_name)]

    # 删除相关权重
    for key in keys_to_remove:
        print(f"Removing {key}")
        del state_dict[key]

    ckpt['model'].load_state_dict(state_dict, strict=False)

    # 保存修改后的模型
    torch.save(ckpt, '../yolo8_modified.pt')

def compare_c4(remove_model_prefix=False, get_same=False):
    model_a_state_dict = torch.load('ckpt_mseg102.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model_b_state_dict = torch.load('ckpt_mseg1022.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

    if remove_model_prefix:
        model_a_state_dict = {
            k.replace('model.', ''): v
            for k, v in model_a_state_dict.items()
        }
        model_b_state_dict = {
            k.replace('model.', ''): v
            for k, v in model_b_state_dict.items()
        }

    # 初始化存储不一致的层
    inconsistent_layers = []
    consistent_layers = []
    # 比较两个模型的参数
    for (name_a, param_a), (name_b, param_b) in zip(model_a_state_dict.items(), model_b_state_dict.items()):
        if name_a != name_b:
            print(f"{name_a} vs {name_b}")
        else:
            # 检查参数是否一致
            if not torch.equal(param_a, param_b):
                inconsistent_layers.append([name_a, param_a.shape, param_b.shape])
            else:
                consistent_layers.append(name_a)

    # 输出不一致的层
    if get_same:
        if consistent_layers:
            print("consistent layers found:")
            for layer in consistent_layers:
                print(layer)
            print(f'total length: {len(consistent_layers)} in {len(model_a_state_dict)}')
        else:
            print("All layers are inconsistent.")
    else:
        if inconsistent_layers:
            print("Inconsistent layers found:")
            for layer in inconsistent_layers:
                print(layer)
            print(f'total length: {len(inconsistent_layers)} in {len(model_a_state_dict)}')
        else:
            print("All layers are consistent.")


def compare_mseg_seg_ckpt():
    model_a_state_dict = torch.load('ckpt_seg1001.pth')
    model_b_state_dict = torch.load('ckpt_mseg1001.pth')
    # 初始化存储不一致的层
    inconsistent_layers = []

    # 比较两个模型的参数
    for (name_a, param_a), (name_b, param_b) in zip(model_a_state_dict.items(), model_b_state_dict.items()):
        if name_a != name_b:
            print(f"{name_a} vs {name_b}")
        else:
            # 检查参数是否一致
            if not torch.equal(param_a, param_b):
                inconsistent_layers.append([name_a, param_a.shape, param_b.shape])

    # 输出不一致的层
    if inconsistent_layers:
        print("Inconsistent layers found:")
        for layer in inconsistent_layers:
            print(layer)
        print(f'total length: {len(inconsistent_layers)} in {len(model_a_state_dict)}')
    else:
        print("All layers are consistent.")

def compare_mseg_seg_data():
    data_a = torch.load('data_seg1001.pth')['img']
    data_b = torch.load('data_mseg1001.pth')['img']

    result = torch.equal(data_a, data_b)
    print(result)
    # result = torch.equal(data_a[0], data_b[0])
    # print(result)
    # result = torch.equal(data_a[1], data_b[1])
    # print(result)
    # result = torch.equal(data_a[2], data_b[2])
    # print(result)

def compare_mseg_seg_pred():
    pred_a = torch.load('pred_seg2000.pth')
    pred_b = torch.load('pred_mseg2000.pth')

    result = torch.equal(pred_a[0][0], pred_b[0][0][:,:68])
    print(result)
    result = torch.equal(pred_a[0][1], pred_b[0][1][:,:68])
    print(result)
    result = torch.equal(pred_a[0][2], pred_b[0][2][:,:68])
    print(result)
    result = torch.equal(pred_a[1], pred_b[1])
    print(result)
    result = torch.equal(pred_a[2], pred_b[2])
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
    feat_a = torch.load(path_a)
    feat_b = torch.load(path_b)

    result = torch.equal(feat_a, feat_b)
    print(path_a, path_b, result)
    # result = torch.equal(feat_a[0], feat_b[0][:, :68])
    # print(result)
    # result = torch.equal(feat_a[1], feat_b[1][:, :68])
    # print(result)
    # result = torch.equal(feat_a[2], feat_b[2][:, :68])
    # print(result)

if __name__ == '__main__':
    pass
    # remove_c4()
    # compare_c4(True, False)
    # compare_mseg_seg_ckpt()
    # compare_mseg_seg_data()
    # compare_mseg_seg_pred()
    compare_mseg_seg_loss()

    # for i in range(9):
    #     path_a = f'pred_seg201{i}.pth'
    #     path_b = f'pred_mseg201{i}.pth'
    #     compare_mseg_seg_feat(path_a, path_b)