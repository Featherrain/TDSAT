import os
import torch
import logging
from scipy.io import savemat
import models
import numpy as np
from utility import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = [int(str_arg) for str_arg in str_args if int(str_arg) >= 0]
        return parsed_args

    parser.add_argument('--prefix', '-p', type = str, default='temp', help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=False, choices=model_names,
                        help = 'model architecture: ' + ' | '.join(model_names), default='tdsat' )
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=1e-4')
    parser.add_argument('--wd', type=float, default=0, help='weight decay. default = 0')
    parser.add_argument('--loss', type=str, default='l2', choices=['l1','l2','smooth_l1','ssim','l2_ssim'],
                        help='loss')
    parser.add_argument("--task_id", '-t', type=int, default=0)
    parser.add_argument('--beta', '-b', type=float, default=0.95)
    parser.add_argument("--epochs", '-e', type=int, default=-1)
    parser.add_argument('--init', type=str, default='kn',choices=['kn', 'ku', 'xn', 'xu', 'edsr'],
                        help='which init scheme to choose.')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--threads', type=int, default=0,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed to use. default=2018')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')
    parser.add_argument('--slice', action='store_true',
                            help='forward chop')
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--ImportancePath', '-ip', type=str, default=None, help='importance to use')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='./data/ICVL/trainset/ICVL64_31.npz', help='data root')
    parser.add_argument('--valroot', '-v', type=str,
                        default='./data/ICVL/valset_gauss', help='validation set root')
    parser.add_argument('--clip', type=float, default=1e6, help='gradient clip threshold')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids, ex:0,1,2,3')
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt

class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = os.path.join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0 # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        cuda = not self.opt.no_cuda
        self.device = 'cuda' if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch.lower()]()

        """Params Init"""
        init_params(self.net, init_type=self.opt.init)

        if cuda and len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)
        if cuda and len(self.opt.gpu_ids) == 1:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        """Loss Function"""
        self.criterion = init_criterion(self.opt.loss)
        print('criterion: ', self.criterion)

        if cuda:
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            print('==> Loading checkpoint...')
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
            # print(self.net)


    def load(self, resumePath=None, load_opt=True):
        model_best_path = os.path.join(self.basedir, self.prefix, 'model_latest.pth')
        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        #### comment when using memnet
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.get_net().load_state_dict(checkpoint['net'])


    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net

    """Forward Functions"""
    def forward_chop(self, x, base=32):
        if len(x.shape) == 5:
            n, c, b, h, w = x.size()
        else:
            n, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2

        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)

        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)

        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1

        output /= output_w

        return output

    def forward_slice(self, inputs, base = 31):
        if len(inputs.shape) == 5:
            n, c, b, h, w = inputs.size()
        else:
            n, b, h, w = inputs.size()
        output = torch.zeros_like(inputs)
        cnt = torch.zeros_like(inputs)
        for bb in range(0, b-base+1, base):
            x = inputs[...,bb:bb+base,:,:]
            output[...,bb:bb+base,:,:] += self.net(x)
            cnt[...,bb:bb+base,:,:] += 1
        if b % base != 0:
            x = inputs[..., b-base:b, :, :]
            output[...,b-base:b,:,:] += self.net(x)
            cnt[...,b-base:b,:,:] += 1
        output = output / cnt
        return output


    def forward_chop_slice(self, inputs, base=31, win_size=32):
        if len(inputs.shape) == 5:
            n, c, b, h, w = inputs.size()
            output = torch.zeros_like(inputs)
            cnt = torch.zeros_like(inputs)
            for bb in range(b - base + 1):
                x = inputs[:, :, bb:bb + base, :, :]
                if h % win_size != 0 or w % win_size != 0:
                    output[:, :, bb:bb + base, :, :] += self.forward_chop(x, win_size)
                else:
                    output[:, :, bb:bb + base, :, :] += self.net(x)
                cnt[:, :, bb:bb + base, :, :] += 1
            output = output / cnt
        else:
            n, b, h, w = inputs.size()
            output = torch.zeros_like(inputs)
            cnt = torch.zeros_like(inputs)
            for bb in range(b - base + 1):
                x = inputs[:, bb:bb + base, :, :]
                if h % win_size != 0 or w % win_size != 0:
                    output[:, bb:bb + base, :, :] += self.forward_chop(x, win_size)
                else:
                    output[:, bb:bb + base, :, :] += self.net(x)
                cnt[:, bb:bb + base, :, :] += 1
            output = output / cnt

        return output



    def forward(self, inputs):
        if self.opt.chop and not self.opt.slice:
            output = self.forward_chop(inputs)
        elif not self.opt.chop and self.opt.slice:
            output = self.forward_slice(inputs)
        elif self.opt.chop and self.opt.slice:
            output = self.forward_chop_slice(inputs)
        else:
            output = self.net(inputs)
        return output

    def __step(self, train, inputs, targets):
        if train:
            self.optimizer.zero_grad()
            self.net.zero_grad()
        loss_data = 0
        total_norm = None
        time_start = time.time()
        if self.get_net().bandwise:
            O = []
            for ti, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                if self.net._get_name() == 'HSID':
                    o = self.forward((i, inputs, ti))
                else:
                    o = self.forward(i)
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs = torch.cat(O, dim=1)
            time_end = time.time()
        else:
            outputs = self.forward(inputs)
            time_end = time.time()
            loss = self.criterion(outputs, targets)
            if train:
                loss.backward()
            loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)#
            self.optimizer.step()
        timecost = time_end - time_start
        return outputs, loss_data, total_norm, timecost

    """Training Functions"""
    def train(self, train_loader):

        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0
        avg_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)


            outputs, loss_data, total_norm, time_cost = self.__step(True, inputs, targets)

            train_loss += loss_data
            avg_loss = train_loss / (batch_idx + 1)

            if not self.opt.no_log:
                self.writer.add_scalar(
                    os.path.join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    os.path.join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            psnr = np.mean(cal_bwpsnr(outputs, targets))
            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4f | Loss: %.4f | Norm: %.4f | PSNR: %.4f '% (avg_loss, loss_data, total_norm, psnr))


        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                os.path.join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)


    """Validation Functions"""
    def validate(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        print('[i] Eval dataset {}...'.format(name))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, loss_data, _, time_cost= self.__step(False, inputs, targets)
                psnr = np.mean(cal_bwpsnr(outputs, targets))
                # Visualize3D(outputs[0].detach().cpu().numpy())
                # Visualize3D(targets[0].detach().cpu().numpy())
                validate_loss += loss_data
                avg_loss = validate_loss / (batch_idx + 1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx + 1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4f | PSNR: %.4f'
                             % (avg_loss, avg_psnr))

        if not self.opt.no_log:
            self.writer.add_scalar(
                os.path.join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                os.path.join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    """Testing Functions"""
    def test(self, test_loader, filename=None, savedir=None, saveimgdir=None, verbose=True):
        #attention! only one sample per batch
        self.net.eval()
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 5))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, _, time_cost = self.__step(False, inputs, targets) # 调用__step
                res_arr[batch_idx, 3] = loss_data
                res_arr[batch_idx, 4] = time_cost
                res_arr[batch_idx, :3] = MSIQA(outputs, targets)
                psnr = res_arr[batch_idx, 0]
                ssim = res_arr[batch_idx, 1]
                sam = res_arr[batch_idx, 2]
                if verbose:
                    print("Batch = %d，PSNR = %.4f, SSIM = %.4f，SAM = %.4f , loss = %.4f ， time_cost== %.4f" % (batch_idx, psnr, ssim, sam, loss_data, time_cost))

                if saveimgdir:
                    filedir = os.path.join(saveimgdir, os.path.basename(dataset.filenames[batch_idx]).split('.')[0])
                    outpath = os.path.join(filedir, '{}.mat'.format(self.opt.arch))
                    if not os.path.exists(filedir):
                        os.makedirs(filedir)
                    if not os.path.exists(outpath):
                        savemat(outpath, {'R_hsi': torch2numpy(outputs, self.net.use_2dconv)})

        if savedir:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            filepath = os.path.join(savedir, filename)
            suffix = filepath.split('.')[-1]
            if suffix == 'npy':
                np.save(filepath, res_arr)
            elif suffix == 'txt':
                np.savetxt(filepath, res_arr)
            else:
                raise 'invalid file format'
        if verbose:
            print((res_arr.mean(0)).tolist())
        return res_arr

    def image_denosing(self, test_loader, saveimgdir):
        #attention! only one sample per batch
        self.net.eval()
        dataset = test_loader.dataset.dataset
        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs = inputs.cuda()
                if self.get_net().bandwise:
                    O = []
                    for time, i in enumerate(inputs.split(1, 1)):
                        if self.net._get_name() == 'HSID':
                            o = self.forward((i, inputs, time))
                        else:
                            o = self.forward(i)
                        O.append(o)
                    outputs = torch.cat(O, dim=1)
                else:
                    outputs = self.forward(inputs)

                """Visualization"""
                input_np = inputs[0].cpu().numpy()
                output_np = outputs[0].cpu().numpy()
                display = np.concatenate([input_np, output_np], axis=-1)
                # Visualize3D(display)
                if len(output_np.shape) == 4:
                    output_np = output_np.squeeze(0)
                if saveimgdir:
                    R_hsi = output_np.transpose((1, 2, 0))
                    dirpath = os.path.join(saveimgdir, os.path.basename(dataset.filenames[batch_idx]).split('.')[0])
                    if not os.path.exists(dirpath):
                        os.makedirs(dirpath)
                    savepath = os.path.join(dirpath,
                                    self.opt.arch + '.mat')
                    savemat(savepath, {'R_hsi': R_hsi})



    """Model Saving Functions"""
    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = os.path.join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }

        state.update(kwargs)

        if not os.path.isdir(os.path.join(self.basedir, self.prefix)):
            os.makedirs(os.path.join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))