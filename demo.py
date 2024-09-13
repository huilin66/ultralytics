import torch
w1_path = "runs/3.pt"
w2_path = "runs/4.pt"
weights1 = torch.load(w1_path)
weights2 = torch.load(w2_path)
print(w1_path)
print(w2_path)
# 比较两个权重
def compare_weights(weights1, weights2):

    for key in weights1.keys():
        if not torch.equal(weights1[key], weights2[key]):
            print(f"[different] {key}")
        else:
            pass
            # print(f"same {key}")

compare_weights(weights1, weights2)

# import torch
# import numpy as np
# import random
# from torch import nn
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#
# # 设置随机种子
# seed = 42
# set_seed(seed)
#
# # 示例：定义一个简单的神经网络并初始化权重
# class SimpleNN(torch.nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc = torch.nn.Linear(10, 5)
#         self._weight_init()
#
#     def forward(self, x):
#         return self.fc(x)
#
#     def _weight_init(self):
#         dd = self.state_dict()
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # nn.init.zeros_(m.weight)  # He 初始化
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#                 print(m)
#                 print(m)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)
#
# # 创建网络实例
# model = SimpleNN()
#
# # 打印权重以验证一致性
# print("Initial weights of the linear layer:")
# print(model.fc.weight)
#
# model2 = SimpleNN()
#
# # 打印权重以验证一致性
# print("Initial weights of the linear layer:")
# print(model2.fc.weight)