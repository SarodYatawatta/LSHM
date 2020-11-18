from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py
import torch.fft

# Train autoencoder and k-harmonic mean clustering using LOFAR data

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

#torch.manual_seed(69)
default_batch=20 # no. of baselines per iter, batch size determined by how many patches are created
num_epochs=40 # total epochs
Niter=20 # how many minibatches are considered for an epoch
save_model=True
load_model=False

# file names have to match the SAP ids in the sap_list
file_list=['/home/sarod/L785751.MS_extract.h5','/home/sarod/L785751.MS_extract.h5',
   '/home/sarod/L785747.MS_extract.h5', '/home/sarod/L785757.MS_extract.h5',
   '/home/sarod/L696315.MS_extract.h5', '/home/sarod/L696315.MS_extract.h5']
sap_list=['1','2','0','0','1','2']
#file_list=['../../drive/My Drive/Colab Notebooks/L785751.MS_extract.h5','../../drive/My Drive/Colab Notebooks/L785751.MS_extract.h5',
#    '../../drive/My Drive/Colab Notebooks/L785747.MS_extract.h5','../../drive/My Drive/Colab Notebooks/L785757.MS_extract.h5']
#sap_list=['1','2','0','0']


L=256 # latent dimension in real space
Lf=64 # latent dimension in Fourier space
Kc=10 # clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm
alpha=0.1 # loss+alpha*cluster_loss
beta=1.0 # loss+beta*cluster_similarity (penalty)
gamma=0.1 # loss+gamma*augmentation_loss


from lofar_models import *

# patch size of images
patch_size=128

num_in_channels=4 # real,imag XX,YY
# for 32x32 patches
#net=AutoEncoderCNN(latent_dim=L,K=Kc,channels=num_in_channels).to(mydevice)
# for 64x64 patches
#net=AutoEncoderCNN1(latent_dim=L,K=Kc,channels=num_in_channels).to(mydevice)
# for 128x128 patches
net=AutoEncoderCNN2(latent_dim=L,channels=num_in_channels).to(mydevice)
# fft: real,imag, so increase number of channels
fnet=AutoEncoderCNN2(latent_dim=Lf,channels=2*num_in_channels).to(mydevice)
mod=Kmeans(latent_dim=(L+Lf),K=Kc,p=Khp).to(mydevice)

if load_model:
  checkpoint=torch.load('./net.model',map_location=mydevice)
  net.load_state_dict(checkpoint['model_state_dict'])
  net.train()
  checkpoint=torch.load('./fnet.model',map_location=mydevice)
  fnet.load_state_dict(checkpoint['model_state_dict'])
  fnet.train()
  checkpoint=torch.load('./khm.model',map_location=mydevice)
  mod.load_state_dict(checkpoint['model_state_dict'])
  mod.train()

params=list(net.parameters())
params.extend(list(fnet.parameters()))
params.extend(list(mod.parameters()))

import torch.optim as optim
from lbfgsnew import LBFGSNew # custom optimizer
criterion=nn.MSELoss(reduction='sum')
#optimizer=optim.SGD(params, lr=0.001, momentum=0.9)
optimizer=optim.Adam(params, lr=0.001)
#optimizer = LBFGSNew(params, history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)

############################################################
# Augmented loss function
def augmented_loss(mu,batches_per_bline,batch_size):
 # process each 'batches_per_bline' rows of mu
 # total rows : batches_per_bline x batch_size
 loss=0
 for ck in range(batch_size):
   Z=mu[ck*batch_per_bline:(ck+1)*batch_per_bline,:]
   prod=0
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
        fftx=torch.fft.fftn(x-xhat,dim=(2,3))
        y=torch.cat((fftx.real,fftx.imag),1)/(patch_size)
        yhat,ymu=fnet(y)
        # normalize all losses by number of dimensions of the tensor input
        loss1=(criterion(xhat,x))/(x.numel())
        loss2=(criterion(yhat,y))/(y.numel())
        Mu=torch.cat((mu,ymu),1)
        kdist=mod.clustering_error(Mu)
        augmentation_loss=augmented_loss(Mu,batch_per_bline,default_batch)
        clus_sim=mod.cluster_similarity()
        loss=loss1+loss2+alpha*kdist+gamma*augmentation_loss+beta*clus_sim
        if loss.requires_grad:
          loss.backward()
          print('%f %f %f %f %f'%(loss1.data.item(),loss2.data.item(),kdist.data.item(),augmentation_loss.data.item(),clus_sim.data.item()))
        return loss

    optimizer.step(closure)
    #print('iter %d/%d loss %f'%(epoch,i,closure().data.item()))





if save_model:
  torch.save({
    'model_state_dict':net.state_dict()
  },'net.model')
  torch.save({
    'model_state_dict':fnet.state_dict()
  },'fnet.model')
  torch.save({
    'model_state_dict':mod.state_dict()
  },'khm.model')
