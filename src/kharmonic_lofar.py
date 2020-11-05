from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

#torch.manual_seed(69)
default_batch=8 # no. of baselines per iter, batch size determined by how many patches are created
num_epochs=10 # total epochs
Niter=40 # how many minibatches are considered for an epoch
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
Kc=15 # clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm
alpha=0.1 # loss+alpha*cluster_loss


from lofar_models import *

# patch size of images
patch_size=64

# for 32x32 patches
#net=AutoEncoderCNN(latent_dim=L,K=Kc,channels=8).to(mydevice)
# for 64x64 patches
net=AutoEncoderCNN1(latent_dim=L,K=Kc,channels=8).to(mydevice)
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


# train network
for epoch in range(num_epochs):
  for i in range(Niter):
    # get the inputs
    patchx,patchy,inputs=get_data_minibatch(file_list,sap_list,batch_size=default_batch,patch_size=patch_size)
    # wrap them in variable
    inputs=Variable(inputs).to(mydevice)
    (nbatch,nchan,nx,ny)=inputs.shape 
    def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        outputs,mu=net(inputs)
        loss=criterion(outputs,inputs)
        kdist=mod(mu)
        loss=loss/(nbatch*nchan)+alpha*kdist
        #print('%f %f'%(kdist.data.item(),loss.data.item()))
        if loss.requires_grad:
          loss.backward()
        return loss

    optimizer.step(closure)
    print('iter %d/%d loss %f'%(epoch,i,closure().data.item()))





if save_model:
  torch.save({
    'model_state_dict':net.state_dict()
  },'net.model')
  torch.save({
    'model_state_dict':mod.state_dict()
  },'khm.model')
