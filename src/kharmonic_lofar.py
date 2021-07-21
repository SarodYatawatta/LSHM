from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py
import torch.fft

# Note: use_cuda=True is set in lofar_models.py, so make sure to change it
# if you change it in this script as well
from lofar_models import *
# Train autoencoder and k-harmonic mean clustering using LOFAR data

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

#torch.manual_seed(69)
default_batch=30 # no. of baselines per iter, batch size determined by how many patches are created
num_epochs=1 # total epochs
Niter=40 # how many minibatches are considered for an epoch
save_model=True
load_model=True

# scan directory to get valid datasets
# file names have to match the SAP ids in the sap_list
file_list,sap_list=get_fileSAP('/home/sarod')
# or ../../drive/My Drive/Colab Notebooks/

L=256 # latent dimension in real space
Lt=32 # latent dimensions in time/frequency axes (1D CNN)
Kc=10 # clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm
alpha=0.1 # loss+alpha*cluster_loss
beta=0.1 # loss+beta*cluster_similarity (penalty)
gamma=0.1 # loss+gamma*augmentation_loss

# patch size of images
patch_size=128

num_in_channels=4 # real,imag XX,YY

# for 128x128 patches
net=AutoEncoderCNN2(latent_dim=L,channels=num_in_channels).to(mydevice)
# 1D autoencoders
netT=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels).to(mydevice)
netF=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels).to(mydevice)
# Kharmonic model
mod=Kmeans(latent_dim=(L+Lt+Lt),K=Kc,p=Khp).to(mydevice)

if load_model:
  checkpoint=torch.load('./net.model',map_location=mydevice)
  net.load_state_dict(checkpoint['model_state_dict'])
  net.train()
  checkpoint=torch.load('./khm.model',map_location=mydevice)
  mod.load_state_dict(checkpoint['model_state_dict'])
  mod.train()
  checkpoint=torch.load('./netT.model',map_location=mydevice)
  netT.load_state_dict(checkpoint['model_state_dict'])
  netT.train()
  checkpoint=torch.load('./netF.model',map_location=mydevice)
  netF.load_state_dict(checkpoint['model_state_dict'])
  netF.train()


import torch.optim as optim
from lbfgsnew import LBFGSNew # custom optimizer
criterion=nn.MSELoss(reduction='sum')
optimizer=optim.Adam(net.parameters(), lr=0.0001)
optimizerT=optim.Adam(netT.parameters(), lr=0.0001)
optimizerF=optim.Adam(netF.parameters(), lr=0.0001)
#optimizerM=optim.Adam(mod.parameters(), lr=0.0001)
optimizerM = LBFGSNew(mod.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)

############################################################
# Augmented loss function
def augmented_loss(mu,batch_per_bline,batch_size):
 # process each 'batches_per_bline' rows of mu
 # total rows : batches_per_bline x batch_size
 loss=torch.Tensor(torch.zeros(1)).to(mydevice)
 for ck in range(batch_size):
   Z=mu[ck*batch_per_bline:(ck+1)*batch_per_bline,:]
   prod=torch.Tensor(torch.zeros(1)).to(mydevice)
   for ci in range(batch_per_bline):
     zi=Z[ci,:]/(torch.norm(Z[ci,:])+1e-6)
     for cj in range(ci+1,batch_per_bline):
       zj=Z[cj,:]/(torch.norm(Z[cj,:])+1e-6)
       prod=prod+torch.exp(-torch.dot(zi,zj))
   loss=loss+prod/batch_per_bline
 return loss/(batch_size*batch_per_bline)
############################################################


# train network
for epoch in range(num_epochs):
  for i in range(Niter):
    # get the inputs
    patchx,patchy,inputs=get_data_minibatch(file_list,sap_list,batch_size=default_batch,patch_size=patch_size,normalize_data=True,num_channels=num_in_channels)
    # wrap them in variable
    x=Variable(inputs).to(mydevice)
    (nbatch,nchan,nx,ny)=inputs.shape 
    # nbatch = patchx x patchy x default_batch
    # i.e., one baseline (per polarization, real,imag) will create patchx x patchy batches
    batch_per_bline=patchx*patchy

    def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        xhat,mu=net(x)
        # pass through 1D CNN
        iy1=torch.flatten(x,start_dim=2,end_dim=3)
        iy2=torch.flatten(torch.transpose(x,2,3),start_dim=2,end_dim=3)
        yyT,yyTmu=netT(iy1)
        yyF,yyFmu=netF(iy2)
 
        # reshape 1D outputs 
        yyT=yyT.view_as(xhat)
        yyF=yyF.view_as(xhat)
        # reconstruction
        xrecon=xhat+yyT+yyF
        # normalize all losses by number of dimensions of the tensor input
        loss1=(criterion(xrecon,x))/(x.numel())
        Mu=torch.cat((mu,yyTmu,yyFmu),1)

        kdist=alpha*mod.clustering_error(Mu)
        clus_sim=beta*mod.cluster_similarity()
        augmentation_loss=gamma*augmented_loss(Mu,batch_per_bline,default_batch)
        loss=loss1+kdist+augmentation_loss+clus_sim

        if loss.requires_grad:
          loss.backward(retain_graph=True)
          print('%d %d %f %f %f %f'%(epoch,i,loss1.data.item(),kdist.data.item(),augmentation_loss.data.item(),clus_sim.data.item()))
        return loss

    #update parameters
    optimizer.step(closure)
    optimizerT.step(closure)
    optimizerF.step(closure)
    optimizerM.step(closure)

if save_model:
  torch.save({
    'model_state_dict':net.state_dict()
  },'net.model')
  torch.save({
    'model_state_dict':mod.state_dict()
  },'khm.model')
  torch.save({
      'model_state_dict':netT.state_dict()
  },'netT.model')
  torch.save({
      'model_state_dict':netF.state_dict()
  },'netF.model')
