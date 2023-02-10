import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from model.model_SR import *
import args_parser
import h5py
from torch.nn import functional as F
from os.path import exists, join, basename
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
import os
from cal_ssim import SSIM, set_random_seed
from torch.autograd import Variable

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        print(dataset.keys())
        self.GT = dataset.get("GT")
        print(self.GT.shape)
        self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("LRHSI")
        self.RGB = dataset.get("RGB")

    #####必要函数
    def __getitem__(self, index):
        input_rgb = torch.from_numpy(self.RGB[index, :, :, :]).float()
        input_lr = torch.from_numpy(self.LRHSI[index, :, :, :]).float()
        input_lr_u = torch.from_numpy(self.UP[index, :, :, :]).float()
        target = torch.from_numpy(self.GT[index, :, :, :]).float()

        return input_rgb, input_lr, input_lr_u, target
        #####必要函数

    def __len__(self):
        return self.GT.shape[0]

def get_training_set(root_dir):
    train_dir = join(root_dir, "train_cave(with_up)x4.h5")
    return DatasetFromHdf5(train_dir)

def get_val_set(root_dir):
    val_dir = join(root_dir, "validation_cave(with_up)x4.h5")
    return DatasetFromHdf5(val_dir)

opt = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(opt)

def save_checkpoint(model, epoch, t, data):
    model_out_path = "checkpoints/{}_{}_{}/model_epoch_{}.pth.tar".format(opt.arch, data,t,epoch)
    state = {"epoch": epoch, "model": model}

    if not os.path.exists("checkpoints/{}_{}_{}".format(opt.arch, data,t,epoch)):
        os.makedirs("checkpoints/{}_{}_{}".format(opt.arch, data, t,epoch))

    torch.save(state, model_out_path)

    print("Checkpoints saved to {}".format(model_out_path))

def main():
    # load data
    print('===> Loading datasets')
    train_set = get_training_set(opt.dataroot)
    val_set = get_val_set(opt.dataroot)

    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.testBatchSize, shuffle=False)


    if opt.dataset == 'pavia_x4':
        opt.n_bands = 92
        opt.image_size = 64
        opt.n_bands_rgb = 4
    elif opt.dataset == 'cave_x4':
        opt.n_bands = 31
        opt.image_size = 64
        opt.n_bands_rgb = 3
    elif opt.dataset == 'harvard_x4':
        opt.n_bands = 31
        opt.image_size = 64
        opt.n_bands_rgb = 3
    elif opt.dataset == 'harvard_x8':
        opt.n_bands = 31
        opt.n_bands_rgb = 3
        opt.image_size = 64


    # Build the models
    model = PSRTnet(opt).cuda()

    input1 = torch.randn(1, 3, opt.image_size, opt.image_size).cuda()
    input2 = torch.randn(1, 31, opt.image_size, opt.image_size).cuda()

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    print(flop_count_table(FlopCountAnalysis(model, (input1, input2))))

    # Loss and optimizer
    g_ssim = SSIM(size_average=True)
    loss1 = nn.L1Loss().cuda()
    loss2 = g_ssim.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)  ## optimizer 1: AdamW

    # Load the trained model parameters

    if os.path.isfile(opt.model_path):
        print("=> loading checkpoint '{}'".format(opt.model_path))
        checkpoint = torch.load(opt.model_path)
        # print(checkpoint['state_dict'].keys())
        # opt.start_epochs = checkpoint["epoch"] + 1
        #
        # # model = torch.load(opt.model_path)
        # state_dict = checkpoint['state_dict']
        # dict = {}
        # for module in state_dict.items():
        #     k, v = module
        #     if 'model' in k:
        #         k = k.strip('model.')
        #     dict[k] = v
        # checkpoint['state_dict'] = dict
        # model.load_state_dict(checkpoint['state_dict'])
        opt.start_epochs = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.model_path))

    # Epochs
    model.train()
    print ('Start Training: ')
    t = time.strftime("%Y%m%d%H%M")
    for epoch in range(opt.start_epochs, opt.n_epochs+1):
        # One epoch's training
        print ('Train_Epoch_{}: '.format(epoch))
        print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

        for iteration, batch in enumerate(training_data_loader, 1):
            input_rgb, _, input_lr_u, ref = Variable(batch[0]).cuda(), Variable(batch[1]).cuda(), Variable( batch[2]).cuda(), Variable(batch[3], requires_grad=False).cuda()
            out = model(input_rgb, input_lr_u)

            loss_L1 = loss1(out, ref)
            # loss_ssim = loss2(out, ref)

            loss = loss_L1

            optimizer.zero_grad()
            loss.backward()
            # for p in model.parameters():
            #     print(p.grad.norm())

            if opt.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_max_norm)

            optimizer.step()

            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                    loss.item()))

        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(val_data_loader):
                input_rgb, _, input_lr_u, ref = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
                out = model(input_rgb, input_lr_u)
            ref = ref.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            psnr = calc_psnr(ref, out)
            rmse = calc_rmse(ref, out)
            ergas = calc_ergas(ref, out)
            sam = calc_sam(ref, out)
            print('RMSE:   {:.4f};'.format(rmse))
            print('PSNR:   {:.4f};'.format(psnr))
            print('ERGAS:   {:.4f};'.format(ergas))
            print('SAM:   {:.4f}.'.format(sam))
        if epoch % 50 == 0:
            save_checkpoint(model, epoch, t, opt.dataset)


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DIVICES"] = "0"
    set_random_seed(10)
    main()
