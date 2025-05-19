import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms.functional as tf
import glob

dir = '/Users/alex/Documents/MATLAB/cards/imgs_front'
fnames = glob.glob(dir + '/*_s.png')
for fname in fnames:
    # fname = dir + '/0_c.png'
    print(fname)

    im = plt.imread(fname)
    imtorch = torch.from_numpy(im)

    imtorch = imtorch.permute((2, 0, 1))

    it = tf.center_crop(imtorch, [720, 720])

    fn = fname.split('/')[-1]
    it = it.permute((1, 2, 0))

    matplotlib.image.imsave(dir+'/crop/'+fn, it.numpy())

    # plt.imshow(it.permute((1, 2, 0)))
    # plt.show()