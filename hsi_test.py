import argparse
import os
import pandas as pd
from hsi_setup import Engine, train_options
from utility import *
from torchvision.transforms import Compose
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':

    """Testing settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    opt.no_log = True
    opt.no_ropt = True

    print(opt)

    model_names = opt.resumePath.split('/')[-1].split('.')[0]
    prefix = opt.resumePath.split('/')[-2]
    savedir = f'./result/{ opt.arch }/{ prefix }/res_{ model_names }'

    """Set Random Status"""
    seed_everywhere(opt.seed)

    cuda = not opt.no_cuda
    opt.no_log = True

    """Setup Engine"""
    engine = Engine(opt)
    print('model params: %.2f' % (sum([t.nelement() for t in engine.net.parameters()]) / 10 ** 6))


    """Eval Testing Datasets"""

    # Your address
    basefolder = './data/ICVL/testset_gauss'




    data = os.path.basename(os.path.normpath(basefolder))
    print('testing %s..................' % data)

    mat_dataset = MatDataFromFolder(basefolder, size=None)
    if engine.net.use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', transform=lambda x:x),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', transform=lambda x:x[:,...][None]),
        ])
    mat_dataset = TransformDataset(mat_dataset, mat_transform)


    mat_loader = DataLoader(
                    mat_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=False
                )
    filename = data + '.npy'

    engine.test(mat_loader, savedir=savedir, filename=filename)

    # '''Collect Result'''
    # prefix = dataset[0].split('_')[0]+'_'+dataset[0].split('_')[1]
    # suffix = ['_30.npy', '_50.npy', '_70.npy', '_blind.npy',
    #           '_noniid.npy', '_stripe.npy', '_deadline.npy', '_impulse.npy', '_mixture.npy']
    # res_table = pd.DataFrame(columns = ['PSNR', 'SSIM', 'SAM', 'Loss', 'Time'])
    # for sfx in suffix:
    #     filepath = os.path.join(savedir, prefix+sfx)
    #     res = np.load(filepath)
    #     res = res.mean(0)
    #     name = prefix + sfx
    #     name = name.replace('.npy', '')
    #     res_table.loc[name] = res
    # res_table.to_csv(os.path.join(savedir, 'result_'+prefix+'.csv'))
