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
num_epochs=40 # total epochs
Niter=40 # how many minibatches are considered for an epoch
save_model=True
load_model=True
# enable this to use 1D CNN along time/freq axes
time_freq_cnn=True
# to use random affine transforms to augment original data
# Do not enable this unless you are sure
transform_data=False

# scan directory to get valid datasets
# file names have to match the SAP ids in the sap_list
file_list,sap_list=get_fileSAP('/home/sarod')
# or ../../drive/My Drive/Colab Notebooks/

L=256 # latent dimension in real space
Lf=64 # latent dimension in Fourier space
Lt=32 # latent dimensions in time/frequency axes (1D CNN)
Kc=10 # clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm
alpha=0.001 # loss+alpha*cluster_loss
beta=0.001 # loss+beta*cluster_similarity (penalty)
gamma=0.001 # loss+gamma*augmentation_loss


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
if time_freq_cnn:
  # 1D autoencoders
  netT=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels)
  netF=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels)
  mod=Kmeans(latent_dim=(L+Lf+Lt+Lt),K=Kc,p=Khp).to(mydevice)
else:
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
  if time_freq_cnn:
    checkpoint=torch.load('./netT.model',map_location=mydevice)
    netT.load_state_dict(checkpoint['model_state_dict'])
    netT.train()
    checkpoint=torch.load('./netF.model',map_location=mydevice)
    netF.load_state_dict(checkpoint['model_state_dict'])
    netF.train()



params=list(net.parameters())
params.extend(list(fnet.parameters()))
params.extend(list(mod.parameters()))
if time_freq_cnn:
  params.extend(list(netT.parameters()))
  params.extend(list(netF.parameters()))


import torch.optim as optim
from lbfgsnew import LBFGSNew # custom optimizer
criterion=nn.MSELoss(reduction='sum')
#optimizer=optim.SGD(params, lr=0.001, momentum=0.9)
optimizer=optim.Adam(params, lr=0.001)
#optimizer = LBFGSNew(params, history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)

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


# Random affine transform to transform data
if transform_data:
  mytransform=transforms.RandomAffine(degrees=(-90,90),translate=None,scale=(1,3),shear=0.1,fill=0,interpolation=transforms.InterpolationMode.NEAREST)
else:
  mytransform=None

# train network
for epoch in range(num_epochs):
  for i in range(Niter):
    # get the inputs
    patchx,patchy,inputs=get_data_minibatch(file_list,sap_list,batch_size=default_batch,patch_size=patch_size,normalize_data=True,num_channels=num_in_channels,transform=mytransform)
    # wrap them in variable
    x=Variable(inputs).to(mydevice)
    (nbatch,nchan,nx,ny)=inputs.shape 
    # is trasform_data=None, nbatch = patchx x patchy x default_batch
    # else, nbatch = 2 x patchx x patchy x default_batch

    # i.e., one baseline (per polarization, real,imag) will create (2) x patchx x patchy batches
    if transform_data:
     batch_per_bline=2*patchx*patchy
    else:
     batch_per_bline=patchx*patchy
    
    def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        xhat,mu=net(x)
        fftx=torch.fft.fftn(x-xhat,dim=(2,3),norm='ortho') # scale 1/sqrt(patch_size^2)
        # fftshift
        freal,fimag=torch_fftshift(fftx.real,fftx.imag)
        y=torch.cat((freal,fimag),1)
        # clamp high values data
        y.clamp_(min=-10,max=10)
        yhat,ymu=fnet(y)

        if time_freq_cnn:
          # form complex tensors for inverse FFT
          yhatc=torch.complex(yhat[:,0:4],yhat[:,4:8])
          yc=torch.complex(freal,fimag)
          yerror=torch.fft.ifftshift(yc-yhatc,dim=(2,3))
          # get IFFT
          iffty=torch.fft.ifftn(yerror,dim=(2,3),norm='ortho') # scale 1/sqrt(patch_size^2)
          # get real part only
          iy=torch.real(iffty)
          # vectorize 2D time freq axes into a vector
          iy1=torch.flatten(iy,start_dim=2,end_dim=3)
          iy2=torch.flatten(torch.transpose(iy,2,3),start_dim=2,end_dim=3)

          # pass through 1D CNN
          yyT,yyTmu=netT(iy1)
          yyF,yyFmu=netF(iy2)

 
        # normalize all losses by number of dimensions of the tensor input
        loss1=(criterion(xhat,x))/(x.numel())
        loss2=(criterion(yhat,y))/(y.numel()/2) # 1/2 because x2 channels
        if time_freq_cnn:
          Mu=torch.cat((mu,ymu,yyTmu,yyFmu),1)
        else:
          Mu=torch.cat((mu,ymu),1)

        kdist=alpha*mod.clustering_error(Mu)
        clus_sim=beta*mod.cluster_similarity()
        augmentation_loss=gamma*augmented_loss(Mu,batch_per_bline,default_batch)
        loss=loss1+loss2+kdist+augmentation_loss+clus_sim

        if time_freq_cnn:
          lossT=(criterion(iy1,yyT))/(iy1.numel())
          lossF=(criterion(iy2,yyF))/(iy2.numel())
          loss+=lossT+lossF

        if loss.requires_grad:
          loss.backward()
          if time_freq_cnn:
            print('%d %d %f %f %f %f %f %f %f'%(epoch,i,loss1.data.item(),loss2.data.item(),lossT.data.item(),lossF.data.item(),kdist.data.item(),augmentation_loss.data.item(),clus_sim.data.item()))
          else:
            print('%d %d %f %f %f %f %f'%(epoch,i,loss1.data.item(),loss2.data.item(),kdist.data.item(),augmentation_loss.data.item(),clus_sim.data.item()))
        return loss

    # local method for offline update of clustering
    def update_clustering():
      with torch.no_grad():
        xhat,mu=net(x)
        fftx=torch.fft.fftn(x-xhat,dim=(2,3),norm='ortho')
        freal,fimag=torch_fftshift(fftx.real,fftx.imag)
        y=torch.cat((freal,fimag),1)
        yhat,ymu=fnet(y)
        Mu=torch.cat((mu,ymu),1)
        err1=(mod.clustering_error(Mu).data.item())
        mod.offline_update(Mu)
        err2=(mod.clustering_error(Mu).data.item())
        print('%e %e'%(err1,err2))
        del xhat,mu,fftx,y,yhat,ymu,Mu

    #update()
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
  if time_freq_cnn:
    torch.save({
      'model_state_dict':netT.state_dict()
    },'netT.model')
    torch.save({
      'model_state_dict':netF.state_dict()
    },'netF.model')
