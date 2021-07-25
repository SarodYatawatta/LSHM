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
default_batch=12 # no. of baselines per iter, batch size determined by how many patches are created
num_epochs=10 # total epochs
Niter=40 # how many minibatches are considered for an epoch
Nadmm=10 # Inner optimization iterations (ADMM)
save_model=True
load_model=True

# scan directory to get valid datasets
# file names have to match the SAP ids in the sap_list
file_list,sap_list=get_fileSAP('/media/sarod')
# or ../../drive/My Drive/Colab Notebooks/

L=256 # latent dimension in real space
Lt=32 # latent dimensions in time/frequency axes (1D CNN)
Kc=10 # clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm
alpha=1.0 # loss+alpha*cluster_loss
beta=1.0 # loss+beta*cluster_similarity (penalty)
gamma=1.0 # loss+gamma*augmentation_loss
rho=1 # ADMM rho

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
# start with empty parameter list
params=list()
#params.extend(list(net.parameters()))
#params.extend(list(netT.parameters()))
#params.extend(list(netF.parameters()))
params.extend(list(mod.parameters()))

#optimizer=optim.Adam(params, lr=0.001)
optimizer = LBFGSNew(params, history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)

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

    # Lagrange multipliers
    y1=torch.zeros(x.numel(),requires_grad=False).to(mydevice)
    y2=torch.zeros(x.numel(),requires_grad=False).to(mydevice)
    y3=torch.zeros(x.numel(),requires_grad=False).to(mydevice)
    for admm in range(Nadmm):
      def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        x1,mu=net(x)
        # residual
        x11=(x-x1)/2
        # pass through 1D CNN
        iy1=torch.flatten(x11,start_dim=2,end_dim=3)
        yyT,yyTmu=netT(iy1)
        # reshape 1D outputs 
        x2=yyT.view_as(x11)

        iy2=torch.flatten(torch.transpose(x11,2,3),start_dim=2,end_dim=3)
        yyF,yyFmu=netF(iy2)
        # reshape 1D outputs 
        x3=torch.transpose(yyF.view_as(x11),2,3)

        # full reconstruction
        xrecon=x1+x2+x3
 
        # normalize all losses by number of dimensions of the tensor input
        # total reconstruction loss
        loss0=(criterion(xrecon,x))/(x.numel())
        # individual losses for each AE
        loss1=(torch.dot(y1,(x-x1).view(-1))+rho/2*criterion(x,x1))/(x.numel())
        loss2=(torch.dot(y2,(x11-x2).view(-1))+rho/2*criterion(x11,x2))/(x.numel())
        loss3=(torch.dot(y3,(x11-x3).view(-1))+rho/2*criterion(x11,x3))/(x.numel())
        Mu=torch.cat((mu,yyTmu,yyFmu),1)

        kdist=alpha*mod.clustering_error(Mu)
        clus_sim=beta*mod.cluster_similarity()
        augmentation_loss=gamma*augmented_loss(Mu,batch_per_bline,default_batch)
        loss=loss0+loss1+loss2+loss3+kdist+augmentation_loss+clus_sim

        if loss.requires_grad:
          loss.backward(retain_graph=True)
          # output line contains:
          # epoch batch admm total_loss loss_AE1 loss_AE2 loss_AE3 loss_KHarmonic loss_augmentation loss_similarity
          print('%d %d %d %f %f %f %f %f %f %f'%(epoch,i,admm,loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),kdist.data.item(),augmentation_loss.data.item(),clus_sim.data.item()))
        return loss

      #update parameters
      optimizer.step(closure)
      # update Lagrange multipliers
      with torch.no_grad():
        x1,_=net(x)
        x11=(x-x1)/2
        iy1=torch.flatten(x11,start_dim=2,end_dim=3)
        yyT,_=netT(iy1)
        # reshape 1D outputs 
        x2=yyT.view_as(x11)

        iy2=torch.flatten(torch.transpose(x11,2,3),start_dim=2,end_dim=3)
        yyF,_=netF(iy2)
        # reshape 1D outputs 
        x3=torch.transpose(yyF.view_as(x11),2,3)

        y1=y1+rho*(x-x1).view(-1)
        y2=y2+rho*(x11-x2).view(-1)
        y3=y3+rho*(x11-x3).view(-1)
        #print("%d %f %f %f"%(admm,torch.norm(y1),torch.norm(y2),torch.norm(y3)))

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
