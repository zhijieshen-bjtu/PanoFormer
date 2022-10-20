import numpy as np
from PIL import Image
from . import py360convert as py360
import matplotlib.pyplot as plt
import torch



def pole_RMSE(pred, gt):
    b, c, h, w = pred.shape
    pred = pred[0].reshape(w,h,c).cpu().data
    gt = gt[0].reshape(w,h,c).cpu().data
    pred_dict = py360.e2c(pred, face_w=h // 2, mode="bilinear", cube_format='dict')
    gt_dict = py360.e2c(gt, face_w=h // 2, mode="bilinear", cube_format='dict')
    pred_UD = torch.cat((torch.tensor(pred_dict['U']), torch.tensor(pred_dict['D'])), dim=2)
    gt_UD = torch.cat((torch.tensor(gt_dict['U']), torch.tensor(gt_dict['D'])), dim=2)
    pole_RMSE = (pred_UD - gt_UD)**2
    pole_RMSE = torch.sqrt(pole_RMSE.mean())

    return pole_RMSE




# img = np.array(Image.open('test_0.jpg'))
#
# img = torch.tensor(img)
#
# cube_dict = py360.e2c(img, face_w=128, mode="bilinear", cube_format='dict')
#
# plt.imshow(cube_dict['U'])
#
#
#
# print('cube_dict["F"].shape:', cube_dict["F"].shape)

# You can make convertion between supported cubemap format
# cube_h = py360convert.cube_dice2h(cube_dice)  # the inverse is cube_h2dice
# cube_dict = py360convert.cube_h2dict(cube_h)  # the inverse is cube_dict2h
# cube_list = py360convert.cube_h2list(cube_h)  # the inverse is cube_list2h
# print('cube_dice.shape:', cube_dice.shape)
# print('cube_h.shape:', cube_h.shape)
# print('cube_dict.keys():', cube_dict.keys())
# print('cube_dict["F"].shape:', cube_dict["F"].shape)
# print('len(cube_list):', len(cube_list))
# print('cube_list[0].shape:', cube_list[0].shape)