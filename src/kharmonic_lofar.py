from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py

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
file_list=['/home/sarod/L785751.MS_extract.h5',
   '/home/sarod/L785751.MS_extract.h5']
sap_list=['1','2']
#file_list=['../../drive/My Drive/Colab Notebooks/L785751.MS_extract.h5','../../drive/My Drive/Colab Notebooks/L785751.MS_extract.h5',
#    '../../drive/My Drive/Colab Notebooks/L785747.MS_extract.h5','../../drive/My Drive/Colab Notebooks/L785757.MS_extract.h5']
#sap_list=['1','2','0','0']


L=256 # latent dimension
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
net=AutoEncoderCNN2(latent_dim=L,K=Kc,channels=num_in_channels).to(mydevice)
mod=Kmeans(latent_dim=L,K=Kc,p=Khp).to(mydevice)

if load_model:
  checkpoint=torch.load('./net.model',map_location=mydevice)
  net.load_state_dict(checkpoint['model_state_dict'])
  net.train()
  checkpoint=torch.load('./khm.model',map_location=mydevice)
  mod.load_state_dict(checkpoint['model_state_dict'])
  mod.train()

params=list(net.parameters())
params.extend(list(mod.parameters()))

import torch.optim as optim
#from lbfgsnew import LBFGSNew # custom optimizer
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
 return loss/batch_size

def augmented_loss1(mu,batches_per_bline,batch_size):
 # process each 'batches_per_bline' rows of mu
 # total rows : batches_per_bline x batch_size
 loss=0
 for ck in range(batch_size):
   z=mu[ck*batch_per_bline:(ck+1)*batch_per_bline,:]
   Z=torch.matmul(z,torch.transpose(z,0,1))
   numerator=torch.sum(torch.diagonal(Z))
   denominator=0
   for ci in range(batch_per_bline):
     denominator=denominator+torch.sum(Z[ci,ci+1:])/batch_per_bline
   loss=loss+numerator/(denominator+1e-6)
 return loss/batch_size
############################################################

# train network
for epoch in range(num_epochs):
  for i in range(Niter):
    # get the inputs
    patchx,patchy,inputs=get_data_minibatch(file_list,sap_list,batch_size=default_batch,patch_size=patch_size,normalize_data=True,num_channels=num_in_channels)
    # wrap them in variable
    inputs=Variable(inputs).to(mydevice)
    (nbatch,nchan,nx,ny)=inputs.shape 
    # nbatch = patchx x patchy x default_batch
    # i.e., one baseline will create patchx x patchy batches
    batch_per_bline=patchx*patchy
    
    def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        outputs,mu=net(inputs)
        loss=criterion(outputs,inputs)/(nbatch*nchan)
        kdist=mod(mu)
        augmentation_loss=augmented_loss(mu,batch_per_bline,default_batch)
        clus_sim=mod.cluster_similarity()
        loss=loss+alpha*kdist+gamma*augmentation_loss+beta*clus_sim
        if loss.requires_grad:
          loss.backward()
          print('%f %f %f %f'%(loss.data.item(),kdist.data.item(),augmentation_loss.data.item(),clus_sim.data.item()))
        return loss

    optimizer.step(closure)
    #print('iter %d/%d loss %f'%(epoch,i,closure().data.item()))





if save_model:
  torch.save({
    'model_state_dict':net.state_dict()
  },'net.model')
  torch.save({
    'model_state_dict':mod.state_dict()
  },'khm.model')
