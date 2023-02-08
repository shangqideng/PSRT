import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='PSRT')

    parser.add_argument('-root', type=str, default='./data')
    parser.add_argument('--dataroot', type=str, default='F:\Data\HSI\cave_x4')
    parser.add_argument('--dataset', type=str, default='cave_x4')
    parser.add_argument('--n_bands', type=int, default=20)
    parser.add_argument('--clip_max_norm', type=int, default=10)

    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--model_path', type=str, 
                            default='./checkpoints/PSRT_cave_x4_202302081324/model_epoch_0.pth.tar',
                            help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='F:\Data\HSI',
                            help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='F:\Data\HSI',
                            help='directory for resized images')

    # learning settingl
    parser.add_argument('--start_epochs', type=int, default=0,
                        help='end epoch for training')
    parser.add_argument('--n_epochs', type=int, default=2001,
                            help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=64)
 
    args = parser.parse_args()
    return args
 