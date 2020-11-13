from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py


# Load pre-trained model to evaluate clustering for given LOFAR dataset

L=256 # latent dimension
Kc=10 # clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm


from lofar_models import *

num_in_channels=4 # real,imag XX,YY
# 32x32 patches
#net=AutoEncoderCNN(latent_dim=L,K=Kc,channels=8)
# 64x64 patches
#net=AutoEncoderCNN1(latent_dim=L,K=Kc,channels=8)
# for 128x128 patches
net=AutoEncoderCNN2(latent_dim=L,K=Kc,channels=num_in_channels)
mod=Kmeans(latent_dim=L,K=Kc,p=Khp)

checkpoint=torch.load('./net.model',map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['model_state_dict'])
checkpoint=torch.load('./khm.model',map_location=torch.device('cpu'))
mod.load_state_dict(checkpoint['model_state_dict'])


file_list=['/home/sarod/L785751.MS_extract.h5',
   '/home/sarod/L785751.MS_extract.h5']
sap_list=['1','2']


# get nbase,nfreq,ntime,npol,ncomplex
nbase,nfreq,ntime,npol,ncomplex=get_metadata(file_list[0],sap_list[0])
# iterate over each baselines
for nb in range(nbase):
 patchx,patchy,x=get_data_for_baseline(file_list[0],sap_list[0],baseline_id=nb,patch_size=128,num_channels=num_in_channels)
 # get latent variable
 xhat,mu=net(x)
 kdist=mod(mu)
 (nbatch,_)=mu.shape
 dist=torch.zeros(Kc,1)
 for ck in range(Kc):
   for cn in range(nbatch):
     dist[ck]=dist[ck]+(torch.norm(mu[cn,:]-mod.M[ck,:],2))
 dist=dist/nbatch
 print(dist)
 (values,indices)=torch.min(dist,0)
 print('%d %f %d'%(nb,kdist,indices[0])) 
 vis=get_data_for_baseline_flat(file_list[0],sap_list[0],baseline_id=nb,num_channels=num_in_channels)
 torchvision.utils.save_image(vis[0,0].data, 'b'+str(indices[0].data.item())+'_'+str(nb)+'.png')

