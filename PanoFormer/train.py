from __future__ import absolute_import, division, print_function
import os
import argparse

from trainer import Trainer
#from test import Tester

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Training")

# system settings
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=[0], help="available gpus")

# model settings
parser.add_argument("--model_name", type=str, default="panodepth", help="folder to save the model in")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")

# loading and logging settings
parser.add_argument("--load_weights_dir", default='./tmp/panodepth/models/weights_pretrain', type=str, help="folder of model to load")#, default='./tmp_abl_offset/panodepth/models/weights_49'
parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "tmp"), help="log directory")
parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")

# data augmentation settings
parser.add_argument("--disable_color_augmentation", action="store_true", help="if set, do not use color augmentation")
parser.add_argument("--disable_LR_filp_augmentation", action="store_true",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", action="store_true",
                    help="if set, do not use yaw rotation augmentation")


args = parser.parse_args()


def main():
    trainer = Trainer(args)
    trainer.train()
    #tester = Tester(args)
    #tester.test()


if __name__ == "__main__":
    main()
