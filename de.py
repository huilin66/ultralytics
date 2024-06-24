# import torch
#
# # 指定数组大小
# gt_attributes_shape = (1, 14)
# pred_attributes_shape = (300, 14)
# iou_shape = (1, 300)
#
# gt_attributes = torch.randint(0, 2, size=gt_attributes_shape, dtype=torch.int)
# gt_attributes = gt_attributes.bool()
#
#
# iou = torch.rand(size=iou_shape)
#
# pred_attributes = torch.rand(size=pred_attributes_shape)
#
# iou50 = iou > 0.5
# idx = iou50.permute((1, 0)).squeeze(-1)
# print(idx.shape)
#
#
# pred_attributes = pred_attributes[idx]
# print(pred_attributes.shape)
# correct_attributes = gt_attributes == pred_attributes
# print(torch.mean(correct_attributes.float(), axis=0))
# import numpy as np
# x = np.nan
# print(x)
# print(np.isnan(x))
import os.path

print(os.path.isdir('./docs'))
print(os.path.isdir(os.path.join(r'E:\repository\ultralytics', './docs')))
print(os.path.isdir(os.path.join(r'E:\repository\ultralytics', '../docs')))