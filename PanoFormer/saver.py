import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
import open3d as o3d


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass


class Saver(object):

    def __init__(self, save_dir):
        self.idx = 0
        self.save_dir = os.path.join(save_dir, "results")
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)

    def save_as_point_cloud(self, depth, rgb, path, mask=None):
        h, w = depth.shape
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = -np.repeat(Phi, h, axis=0)

        X = depth * np.sin(Theta) * np.sin(Phi)
        Y = depth * np.cos(Theta)
        Z = depth * np.sin(Theta) * np.cos(Phi)

        if mask is None:
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            R = rgb[:, :, 0].flatten()
            G = rgb[:, :, 1].flatten()
            B = rgb[:, :, 2].flatten()
        else:
            X = X[mask]
            Y = Y[mask]
            Z = Z[mask]
            R = rgb[:, :, 0][mask]
            G = rgb[:, :, 1][mask]
            B = rgb[:, :, 2][mask]

        XYZ = np.stack([X, Y, Z], axis=1)
        RGB = np.stack([R, G, B], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(RGB)
        o3d.io.write_point_cloud(path, pcd)

    def save_samples(self, rgbs, gt_depths, pred_depths, depth_masks=None):
        """
        Saves samples
        """
        rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        depth_preds = pred_depths.cpu().numpy()
        gt_depths = gt_depths.cpu().numpy()
        if depth_masks is None:
            depth_masks = gt_depths != 0
        else:
            depth_masks = depth_masks.cpu().numpy()

        for i in range(rgbs.shape[0]):
            self.idx = self.idx+1
            mkdirs(os.path.join(self.save_dir, '%04d'%(self.idx)))

            cmap = plt.get_cmap("rainbow_r")

            depth_pred = cmap(depth_preds[i][0].astype(np.float32)/10)
            depth_pred = np.delete(depth_pred, 3, 2)
            path = os.path.join(self.save_dir, '%04d' % (self.idx) ,'_depth_pred.jpg')
            cv2.imwrite(path, (depth_pred * 255).astype(np.uint8))

            depth_gt = cmap(gt_depths[i][0].astype(np.float32)/10)
            depth_gt = np.delete(depth_gt, 3, 2)
            depth_gt[..., 0][~depth_masks[i][0]] = 0
            depth_gt[..., 1][~depth_masks[i][0]] = 0
            depth_gt[..., 2][~depth_masks[i][0]] = 0
            path = os.path.join(self.save_dir, '%04d' % (self.idx), '_depth_gt.jpg')
            cv2.imwrite(path, (depth_gt * 255).astype(np.uint8))

            path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_pc_pred.ply')
            self.save_as_point_cloud(depth_preds[i][0], rgbs[i], path)

            path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_pc_gt.ply')
            self.save_as_point_cloud(gt_depths[i][0], rgbs[i], path, depth_masks[i][0])

            rgb = (rgbs[i] * 255).astype(np.uint8)
            path = os.path.join(self.save_dir, '%04d'%(self.idx) , '_rgb.jpg')
            cv2.imwrite(path, rgb[:,:,::-1])

