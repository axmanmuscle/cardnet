import numpy as np
import torch, torchvision
from torchvision.transforms import v2
import glob, os
import torchvision.io as tio
import torchvision.transforms as tt

import matplotlib.pyplot as plt
import multiprocessing

def main():
    torch.manual_seed(2023099)
    dataDir = '/Users/alex/Documents/MATLAB/cards/imgs_front/crop'
    fs = glob.glob(dataDir +'/*.png')

    resize = tt.Resize([240, 240], antialias=True)
    # makeAugments(fs[0], resize)
    # return()
    # create a process pool that uses all cpus
    # ma = lambda x: makeAugments(x, resize)
    with multiprocessing.Pool() as pool:
        # call the function for each item in parallel
        pool.map(makeAugments, fs)


def makeAugments(fname):
    resize = tt.Resize([240, 240], antialias=True)
    n = 20
    print(fname)
    policies = [v2.AutoAugmentPolicy.CIFAR10, v2.AutoAugmentPolicy.IMAGENET]
    img = tio.read_image(fname)
    img = img[0:3, :, :]
    fname = fname.split('/')[-1]
    fname = fname.split('.')[0]
    os.makedirs(fname)
    for i in range(len(policies)):
        policy = policies[i]
        augmenter = v2.AutoAugment(policy)
        for j in range(n):
            num = j + i*n
            fnout = fname +'/' + fname +f'{num}.png'
            imgout = resize(augmenter(img))
            # if (torch.norm(img - imgout) < 1e-8):
            #     print('no change')
            #     continue
            # plt.imsave(fnout, imgout.permute((1,2,0)))
            tio.write_png(imgout, fnout)
            # plt.imshow(imgout.permute(1,2,0), cmap='gray')
            # plt.axes('off')
            # plt.savefig(fnout, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    main()