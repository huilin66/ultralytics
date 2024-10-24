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

def compare_c4():
    model_a_state_dict = torch.load('2.pth')
    model_b_state_dict = torch.load('3.pth')

    # 初始化存储不一致的层
    inconsistent_layers = []

    # 比较两个模型的参数
    for (name_a, param_a), (name_b, param_b) in zip(model_a_state_dict.items(), model_b_state_dict.items()):
        if name_a != name_b:
            print(f"{name_a} vs {name_b}")
        else:
            # 检查参数是否一致
            if not torch.equal(param_a, param_b):
                inconsistent_layers.append(name_a)

    # 输出不一致的层
    if inconsistent_layers:
        print("Inconsistent layers found:")
        for layer in inconsistent_layers:
            print(layer)
    else:
        print("All layers are consistent.")


if __name__ == '__main__':
    pass
    # remove_c4()
    compare_c4()