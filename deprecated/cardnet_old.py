# Import torch and torchvision packages
import torch
import torch.autograd as autograd
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from cardData import card_dataset
from torch.utils.data import DataLoader

import time

class CardLinear(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.linear1 = nn.Linear(180*180*3, 128)
      self.linear2 = nn.Linear(128, 64)
      self.linear3 = nn.Linear(64, 64)
      self.linear4 = nn.Linear(64, 52)

      self.relu = nn.ReLU()
      self.softmax = nn.Softmax(dim=1)
      self.sigmoid = nn.Sigmoid()

   def forward(self, x):
      x = torch.flatten(x, 1)

      x = self.linear1(x)
      x = self.relu(x)
      x = self.linear2(x)
      x = self.relu(x)
      x = self.linear3(x)
      x = self.relu(x)
      x = self.linear4(x)
      # x = self.relu(x)
      # x = self.softmax(x)
      # x = self.sigmoid(x)

      return x


class Cardnet(torch.nn.Module):
  
  """
  UNet for MRI image reconstruction
  """
  def __init__(self):
    super().__init__()
    ### input [3, 180, 180] (undersampled)
    self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding='same') #output [64, 180, 180]
    self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding='same') #output [64, 180, 180]

    # input [64, 90, 90] bc of max pooling step
    self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding='same') #output [128, 90, 90]
    self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding='same') #output [128, 90, 90]

    # input [128, 45, 45] bc of max pooling step
    self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding='same') #output [256, 45, 45]
    self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding='same') #output [256, 45, 45]

    ## DECODER
    self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) #output 
    self.dconv11 = nn.Conv2d(128, 128, kernel_size=3, padding='same') #output 
    self.dconv12 = nn.Conv2d(128, 64, kernel_size=3, padding='same') #output

    self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) #output
    self.dconv21 = nn.Conv2d(32, 32, kernel_size=3, padding='same') #output
    self.dconv22 = nn.Conv2d(32, 16, kernel_size=3, padding='same') #output

    self.linear1 = nn.Linear(32400, 64)
    #self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(64, 52)

    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=2)

    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):

    x = self.conv11(x)
    x = self.relu(x)
    x = self.conv12(x)
    x1 = self.relu(x)
    x = self.maxpool(x1)

    x = self.conv21(x)
    x = self.relu(x)
    x = self.conv22(x)
    x2 = self.relu(x)
    x = self.maxpool(x2)

    x = self.conv31(x)
    x = self.relu(x)
    x = self.conv32(x)
    x3 = self.relu(x)
    # print(f'before upconv, x size {x.size()}')
    x = self.upconv1(x)
    # print(f'before concat, x3 size {x3.size()} and x size: {x.size()}')
    # x = torch.concatenate((x, x3), dim=1)
    # print(f'after concat, x is size {x.size()}')

    x = self.dconv11(x)
    # print(f'after dconv11, x is size {x.size()}')
    x = self.relu(x)
    x = self.dconv12(x)
    # print(f'after dconv12, x is size {x.size()}')
    x = self.relu(x)

    x = self.upconv2(x)
    # print(f'before concat, x3 size {x3.size()} and x size: {x.size()}')
    # x = torch.concatenate((x, x2), dim=1)
    # print(f'after concat, x is size {x.size()}')

    x = self.dconv21(x)
    # print(f'after dconv21, x is size {x.size()}')
    x = self.relu(x)
    x = self.dconv22(x)
    # print(f'after dconv22, x is size {x.size()}')
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.maxpool(x)

    x = torch.flatten(x,1)

    x = self.linear1(x)
    #x = self.linear2(x)
    x = self.linear3(x)
    x = self.softmax(x)

    return x
  
def main(myDevice):
  runOnGPU = True

  training_datadir = '/Users/alex/Documents/python/cards/augments_916_3'
  # training_datadir = '/Users/alex/Documents/python/cards/training_work'
  training_dataset = card_dataset(training_datadir)

  testing_datadir = '/Users/alex/Documents/python/cards/full_testing'
  # testing_datadir = '/Users/alex/Documents/python/cards/testing'
  testing_dataset = card_dataset(testing_datadir)

  minibatch_size = 10

  trainloader = DataLoader( training_dataset, batch_size = minibatch_size, shuffle = True)

  testloader = DataLoader( testing_dataset, batch_size = minibatch_size, shuffle=True)

  # net = Cardnet()
  net = CardLinear()
  if runOnGPU:
    net = net.to( myDevice )

  criterion = nn.CrossEntropyLoss()  # Softmax is embedded in loss function
  optimizer = optim.SGD( net.parameters(), lr=0.000000001, momentum=0.9 )

   # Train the network
  startTrainTime = time.time()
  for epoch in range(50):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
      
          # get the inputs; data is a list of [inputs, labels]
          if runOnGPU == True:
              inputs, labels = data[0].to(myDevice), data[1].to(myDevice)
          else:
              inputs, labels = data

          # zero the parameter gradients
          optimizer.zero_grad()

          # print(f'iter {i} true label is {labels}')

          # forward + backward + optimize
          outputs = net(inputs)
          _, predicted = torch.max(outputs.data, 1)
          # print(f'predicted output is {predicted}')
          loss = criterion(outputs, labels)
          # print(f'loss is {loss}')
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 10 == 9:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 9:.3f}')
            running_loss = 0.0

  endTrainTime = time.time()
  trainTime = ( endTrainTime - startTrainTime ) / 60
  print(f'Total training time for {epoch+1} epochs: {trainTime} minutes')

  ## write validation?

  # Determine accuracy on test set
  correct = 0
  total = 0
  class_correct = list(0. for i in range(53))
  class_total = list(0. for i in range(53))
  with torch.no_grad():  # since we're not training, we don't need to calculate the gradients for our outputs
    for i, data in enumerate( testloader ):
      if runOnGPU == True:
        images, labels = data[0].to(myDevice), data[1].to(myDevice)
      else:
        images, labels = data
      outputs = net( Variable(images) )
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum()

      c = (predicted == labels).squeeze()
      for i in range( minibatch_size ):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

  print('Accuracy of the network on the %d test images: %d %%' % ( total, 100 * correct / total))

  return 0

if __name__ == "__main__":
  # Move to faster device if available
    if torch.backends.mps.is_available():
        myDevice = torch.device( "mps" )
    elif torch.cuda.is_available():
        # Note: on AMD GPUs, 'cuda' refers to 'ROCm' and should work seamlessly
        # according to https://docs.amd.com/en-US/bundle/AMD-Radeon-PRO-V620-and-W6800-Support-Guide-v5.1/page/AMD_Radeon_PRO_V620_and_W6800_Machine_Learning.html
        myDevice = torch.device( "cuda" )
    else:
        myDevice = torch.device( "cpu" )
    
    net = CardLinear()
    #summary(net, (1, 3, 180, 180))
    #net = Cardnet()
    # summary(net, (10, 3, 180, 180))
    main(myDevice)