import sys
import threading
import time
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import cv2
from PIL import Image

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

term_width = 30
TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')


    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    L.append(' | Remain: %.3f h' % (step_time*(total-current-1) / 3600))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    sys.stdout.write(' %d/%d ' % (current+1, total))
    sys.stdout.flush()




def torch2numpy(hsi, use_2dconv):
    if use_2dconv:
        R_hsi = hsi.data[0].cpu().numpy().transpose((1, 2, 0))
    else:
        R_hsi = hsi.data[0].cpu().numpy()[0, ...].transpose((1, 2, 0))
    return R_hsi

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def Visualize3D(data, meta=None, frame = 0):
    data = np.squeeze(data)
    for ch in range(data.shape[0]):
        data[ch, ...] = minmax_normalize(data[ch, ...])
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # l = plt.imshow(data[frame,:,:])
    l = plt.imshow(data[frame, :, :], cmap='gray')  # shows 256x256 image, i.e. 0th frame
    # plt.colorbar()
    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, 'Frame', 0, data.shape[0] - 1, valinit=0)
    def update(val):
        frame = int(np.around(sframe.val))
        l.set_data(data[frame, :, :])
        if meta is not None:
            axframe.set_title(meta[frame])
    sframe.on_changed(update)
    plt.show()

class writerlogs:
    def __init__(self):
        self.logs = dict()

    def add_scalar(self, key, value):
        if not key in self.logs.keys():
            self.logs[key] = []
        self.logs[key].append(value)


def visulize_attention_ratio(img, attention_mask, ratio=0.5, cmap="jet"):
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
    img = Image.fromarray(img)
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.show()
