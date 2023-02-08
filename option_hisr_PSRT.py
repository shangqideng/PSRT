import argparse
import os

cfg = Config(hisr_cfg())

script_path = os.path.dirname(os.path.dirname(__file__))

root_dir = script_path.split(cfg.task)[0]
print(root_dir)
parser = argparse.ArgumentParser(description='PyTorch derain Training')
parser.add_argument('--name', default='PSRT', type=str)
parser.add_argument('--test_epoch', default='2001', type=int)

model_path = f'{root_dir}/results/{cfg.task}/PSRT/pavia_x4/AdaTrans/Test/model_2022-10-24-15-43/2001.pth.tar'

parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                    help='path to save model')
# * Training
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_scheduler', default=True, type=bool)
parser.add_argument('-samples_per_gpu', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epochs', default=2200, type=int)
parser.add_argument('--workers_per_gpu', default=0, type=int)
parser.add_argument('--resume',
                    default=model_path,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# * Model and Dataset
parser.add_argument('--arch', '-a', metavar='ARCH', default='AdaTrans', type=str)
parser.add_argument('--dataset', default='pavia_x4', type=str,
                    choices=['cave_x4', 'cave_x4', 'harvard_x4', 'Chikusei_x4', 'pavia_x4'],
                    help="Datasets")
parser.add_argument('--eval', default=True, type=bool,
                    help="performing evalution for patch2entire")
parser.add_argument('--crop_batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=8,
                    help='image2patch, set to model and dataset')


# SRData.py dataset setting
parser.add_argument('--model', default='ipt',
                    help='model name')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                    help='train dataset name')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')

args = parser.parse_args()
args.start_epoch = args.best_epoch = 1
args.experimental_desc = "Test"
cfg.reg = False
cfg.clip_max_norm = 5
cfg.reset_lr = True
cfg.mode = "none"
cfg.merge_args2cfg(args)
print(cfg.pretty_text)


