import numpy as np
import torch
#from skimage.measure import compare_ssim, compare_psnr

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from functools import partial

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

# cal_bwssim = Bandwise(compare_ssim) 指定数据范围
cal_bwssim = Bandwise(partial(compare_ssim, data_range=1))
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)
    return np.mean(np.real(np.arccos(tmp)))


def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    return psnr, ssim, sam

if __name__ == '__main__':
    from scipy.io import loadmat
    import torch
    hsi = loadmat('/home/intern_lpang/ImageDenosing/code/Project/STQ3D/images/gavyam_0823-0933/gavyam_0823-0933.mat')['gt']
    R_hsi = loadmat('/home/intern_lpang/ImageDenosing/code/Project/STQ3D/matlab/Result/icvl_512_30/gavyam_0823-0933/BM4D.mat')['R_hsi']
    print(MSIQA(torch.from_numpy(hsi), torch.from_numpy(R_hsi)))
