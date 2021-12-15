from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py

# Note: use_cuda=True is set in lofar_tools.py, so make sure to change it
# if you change it in this script as well
from lofar_tools import *
from lofar_models import *
# Train autoencoder and k-harmonic mean clustering using LOFAR data

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

#torch.manual_seed(69)
default_batch=128 # no. of baselines per iter, batch size determined by how many patches are created
num_epochs=80 # total epochs
Niter=100 # how many minibatches are considered for an epoch
save_model=True
load_model=True

# scan directory to get valid datasets
# file names have to match the SAP ids in the sap_list
file_list,sap_list=get_fileSAP('/media/sarod')
# or ../../drive/My Drive/Colab Notebooks/

# patch size of images
patch_size=128
num_in_channels=4 # real,imag XX,YY
# so each image as a vector has length L
L=num_in_channels*patch_size*patch_size
# hidden dimension size
M=256

lambda1=0.1
eta=0.1
# for each data sample of B batches, we have
# X = A S 
# with X: L x B, A: L x M, S: M x B
# A: dictionary to learn
# cost f(A,S)=||X-A S||^2
# find S sparse min f(A,S) + lambda1 ||S||_1
# find A min f(A,S) + lambda2 ||A(:,i)||_1

# initialize A with random numbers
A=torch.rand((L,M),requires_grad=False,dtype=torch.float32,device=mydevice)
criterion=nn.MSELoss(reduction='sum')

import torch.optim as optim
from lbfgsnew import LBFGSNew # custom optimizer
# train network
for epoch in range(num_epochs):
  for i in range(Niter):
    # get the inputs
    patchx,patchy,inputs,uvcoords=get_data_minibatch(file_list,sap_list,batch_size=default_batch,patch_size=patch_size,normalize_data=True,num_channels=num_in_channels,uvdist=True)
    # wrap them in variable
    x=Variable(inputs).to(mydevice)
    uv=Variable(uvcoords).to(mydevice)
    (nbatch,nchan,nx,ny)=inputs.shape 
    # nbatch = patchx x patchy x default_batch
    # i.e., one baseline (per polarization, real,imag) will create patchx x patchy batches
    batch_per_bline=patchx*patchy

    X=torch.transpose(x.view(-1,L),0,1)
    # setup S for this data batch
    S=torch.rand((M,nbatch),requires_grad=True,dtype=torch.float32,device=mydevice)
    # setup optimizer
    optimizer=LBFGSNew([S],history_size=7,max_iter=10,line_search_fn=True,batch_mode=True)
    def closure():
     if torch.is_grad_enabled():
       optimizer.zero_grad()
     # loss
     loss=criterion(X,torch.matmul(A,S))/(nbatch*L)+lambda1*torch.linalg.norm(S,1)/S.numel()
     if loss.requires_grad:
       #print('%d %d %e'%(epoch,i,loss.data.item()))
       loss.backward()
     return loss

    optimizer.step(closure)

    with torch.no_grad():
      # now update A
      E=X-torch.matmul(A,S)
      dA=torch.zeros((L,M),device=mydevice)
      for ci in range(nbatch):
         dA += torch.outer(E[:,ci],S[:,ci])
      dA /= nbatch
    
      A += eta*dA
      print('A %d %d %e'%(epoch,i,torch.linalg.norm(dA)))


# save columns of A
for ci in range(M):
 Ai=A[:,ci].view(num_in_channels,patch_size,patch_size)
 Ci=channel_to_rgb(Ai)
 torchvision.utils.save_image(Ci,'Ai'+str(ci)+'.png')
