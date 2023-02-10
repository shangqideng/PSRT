import numpy as np
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import scipy.io as sio
import h5py
from os.path import exists, join, basename
import torch.utils.data as data
from model.model_SR import *

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        print(dataset.keys())
        self.GT = dataset.get("GT")
        self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("LRHSI")
        self.RGB = dataset.get("RGB")

    #####必要函数
    def __getitem__(self, index):
        input_pan = torch.from_numpy(self.RGB[index, :, :, :]).float()
        input_lr = torch.from_numpy(self.LRHSI[index, :, :, :]).float()
        input_lr_u = torch.from_numpy(self.UP[index, :, :, :]).float()
        target = torch.from_numpy(self.GT[index, :, :, :]).float()

        return input_pan, input_lr, input_lr_u, target
        #####必要函数

    def __len__(self):
        return self.GT.shape[0]

def get_test_set(root_dir):
    train_dir = join(root_dir, "test_cavepatches256-2.h5")
    return DatasetFromHdf5(train_dir)

from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--dataset', type=str, default='F:\Data\HSI\cave_x4')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--n_bands', type=int, default=31)
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--model_path', type=str,
                    default='./checkpoints/PSRT_cave_x4_202302081324/model_epoch_0.pth.tar',
                    help='path for trained encoder')
opt = parser.parse_args()

test_set = get_test_set(opt.dataset)
test_data_loader = DataLoader(dataset=test_set,  batch_size=opt.testBatchSize, shuffle=False)



def test(test_data_loader):
    model = PSRTnet(opt).cuda()
    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint["model"].state_dict())
    model.eval()
    output = np.zeros((11, opt.image_size, opt.image_size, opt.n_bands))

    psnr_list = []
    sam_list = []
    ergas_list = []

    for index, batch in enumerate(test_data_loader):
        input_rgb, _, input_lr_u, ref = Variable(batch[0]).cuda(), Variable(batch[1],).cuda(), Variable(batch[2]).cuda(), Variable(batch[3]).cuda()

        # ref = ref.cuda()
        out = model(input_rgb, input_lr_u)

        ref = ref.detach().cpu().numpy()
        out1 = out.detach().cpu().numpy()
        psnr = calc_psnr(ref, out1)
        sam = calc_sam(ref, out1)
        ergas = calc_ergas(ref, out1)
        psnr_list.append(psnr)
        sam_list.append(sam)
        ergas_list.append(ergas)

        # rmse = calc_rmse(ref, out)
        # ergas = calc_ergas(ref, out)
        # sam = calc_sam(ref, out)
        # print('RMSE:   {:.4f};'.format(rmse))
        # print('PSNR:   {:.4f};'.format(psnr))
        # print('ERGAS:   {:.4f};'.format(ergas))
        # print('SAM:   {:.4f}.'.format(sam))

        output[index, :, :, :] = out.permute(0, 2, 3, 1).cpu().detach().numpy()
    print('PSNR:   {:.4f};'.format(np.array(psnr_list).mean()))
    print('SAM:   {:.4f};'.format(np.array(sam_list).mean()))
    print('ERGAS:   {:.4f};'.format(np.array(ergas_list).mean()))
    sio.savemat('cave11-psrt.mat', {'output': output})



#
# if not os.path.exists(image_path):
#     os.makedirs(image_path)


test(test_data_loader)











