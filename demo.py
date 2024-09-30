import torch

def remove_c4():
    w1_path = r'runs/mdetect/exp_yolo10x_m34/weights/best.pt'
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
    torch.save(ckpt, 'yolo8_modified.pt')

def compare_c4():
    w1_path = r'runs/mdetect/exp_yolo10x_m34/weights/best.pt'
    ckpt1 = torch.load(w1_path)
    state_dict1 = ckpt1['model'].state_dict()
    w2_path = r'runs/mdetect/exp_yolo10x_m34/weights/best.pt'
    ckpt2 = torch.load(w2_path)
    state_dict2 = ckpt1['model'].state_dict()
    print()

if __name__ == '__main__':
    pass
    # remove_c4()
    compare_c4()