import torch




def continuity(x, gt):

    s = x[:, :, 0, :]
    e = x[:, :, -1, :]
    s_gt = gt[:, :, 0, :]
    e_gt = gt[:, :, -1, :]
    diff = torch.abs((s-e)-(s_gt-e_gt)).mean()
    return diff

#     splitsize = 1024 // 8
#
#     erptemp = x[:, :splitsize, :]
#
#     erptcat = np.concatenate((x,erptemp),axis=1)
#
#     return erptcat[:, splitsize:, :]
#
# path = 'test_72_depth.jpg'
#
# img = cv2.imread(path, cv2.IMREAD_COLOR)
#
# img_np = np.array(img)
#
#
# result = SourseTotrans45_list(img_np)
#
# cv2.imwrite('crop'+path, result)





