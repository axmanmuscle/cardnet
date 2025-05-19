
# This file creates a PyTorch dataset class for playing cards

import numpy as np
import os, re, glob
import torch
import torchvision.transforms.functional as tf
import torchvision.transforms as tt
import torchvision.io as tio

from matplotlib import pyplot as plt


class card_dataset( torch.utils.data.Dataset ):

  def __init__( self, dataDir, size = [180, 180] ):
    self.dataDir = dataDir
    f = glob.glob(dataDir +'/*/*.png')
    self.cardFiles = [os.path.abspath(i) for i in f]
    self.transform = tt.Compose([tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                 tt.Resize(size, antialias=True)])

  def __len__( self ):
    return len(self.cardFiles)
  
  def cat2num( self, cat):
    sc = cat.split('_')
    card = sc[0]
    suit = sc[1]

    if suit == 'h':
      s = 0
    elif suit == 'c':
      s = 1
    elif suit == 'd':
      s = 2
    elif suit == 's':
      s = 3
    else:
      print('something wrong with cardData/cat2num')
      return
    
    if card == 'k':
      c = 13
    elif card == 'q':
      c = 12
    elif card == 'j':
      c = 11
    elif card == '0':
      c = 10
    elif card == 'a':
      c = 1
    else:
      c = int(card)
    
    return c + 13*s


  def __getitem__( self, indx ):
    
    fname = self.cardFiles[indx]
    ls1 = fname.split('/')
    fname1 = ls1[-1]
    card_cat = ls1[-2]
    # print(f'in __getitem__, cat: {card_cat}')

    cat = self.cat2num(card_cat)

    card = tio.read_image(fname)
    card = card.float()
    card = self.transform(card)
    
    return card, cat

if __name__ == "__main__":
  dataDir = '/Users/alex/Documents/python/cards/augments_916_2'

  cardset = card_dataset( dataDir )

  print( 'card data len: {}'.format( cardset.__len__() ))

  for i in [n*41 for n in range(51)]:
    data, label = cardset.__getitem__(i)
    print(f'card label: {label}')

  # plt.imshow(data.permute((1, 2, 0)))
  # plt.show()

  # img = torch.abs( torch.fft.fftshift( torch.fft.ifft2( torch.fft.ifftshift( slice_kspace ) ) ) )
  # plt.imshow( img, cmap='gray' )
  # plt.show()

  print( 'Program ended' )
