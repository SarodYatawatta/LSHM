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

########################################################
def get_data_minibatch(file_list,SAP_list,batch_size=2,patch_size=32):
  # len(file_list)==len(SAP_list)
  # SAP_list should match each file name in file_list
  # open LOFAR H5 file, read data from a SAP,
  # randomly select number of baselines equal to batch_size
  # and sample patches and return input for training
  assert(len(file_list)==len(SAP_list))
  file_id=np.random.randint(0,len(file_list))
  filename=file_list[file_id]
  SAP=SAP_list[file_id]

  # randomly select a file and corresponding SAP
  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']
  # scale factors for the dataset (float32)
  h=f['measurement']['saps'][SAP]['visibility_scale_factors']

  (nbase,ntime,nfreq,npol,ncomplex)=g.shape
  # h shape : nbase, nfreq, npol

  x=torch.zeros(batch_size,8,ntime,nfreq).to(mydevice,non_blocking=True)
  # randomly select baseline subset
  baselinelist=np.random.randint(0,nbase,batch_size)

  ck=0
  for mybase in baselinelist:
   # this is 8 channels in torch tensor
   for ci in range(4):
    # get visibility scales
    scalefac=torch.from_numpy(h[mybase,:,ci]).to(mydevice,non_blocking=True)
    # add missing (time) dimension
    scalefac=scalefac[None,:]
    x[ck,2*ci]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[ck,2*ci]=x[ck,2*ci]*scalefac
    x[ck,2*ci+1]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[ck,2*ci+1]=x[ck,2*ci+1]*scalefac
   ck=ck+1


  #torchvision.utils.save_image(x[0,0].data, 'sample.png')
  stride = patch_size//2 # patch stride (with 1/2 overlap)
  num_channels=8 # 4x2 polarizationx(real,imag)
  y = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
  # get new shape
  (nbase1,nchan1,patchx,patchy,nx,ny)=y.shape
  # create a new tensor
  y1=torch.zeros([nbase1*patchx*patchy,nchan1,nx,ny]).to(mydevice,non_blocking=True)

  # copy data ordered according to the patches
  ck=0
  for ci in range(patchx):
   for cj in range(patchy):
     y1[ck*nbase1:(ck+1)*nbase1,:,:,:]=y[:,:,ci,cj,:,:]
     ck=ck+1

  y = y1
  del x,y1
  # note: nbatch = batch_size x patchx x patchy
  #(nbatch,nchan,nxx,nyy)=y.shape

  # do some rough cleanup of data
  ##y[y!=y]=0 # set NaN,Inf to zero
  torch.clamp(y,-1e6,1e6) # clip high values

  # normalize data
  ymean=y.mean()
  ystd=y.std()
  y.sub_(ymean).div_(ystd)

  return patchx,patchy,y

########################################################
def get_data_for_baseline(filename,SAP,baseline_id,patch_size=32):
  # open LOFAR H5 file, read data from a SAP,
  # return data for given baseline_id

  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']
  # scale factors for the dataset (float32)
  h=f['measurement']['saps'][SAP]['visibility_scale_factors']

  (nbase,ntime,nfreq,npol,ncomplex)=g.shape
  # h shape : nbase, nfreq, npol

  x=torch.zeros(1,8,ntime,nfreq)
  
  mybase=baseline_id
  # this is 8 channels in torch tensor
  for ci in range(4):
    # get visibility scales
    scalefac=torch.from_numpy(h[mybase,:,ci])
    # add missing (time) dimension
    scalefac=scalefac[None,:]
    x[0,2*ci]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[0,2*ci]=x[0,2*ci]*scalefac
    x[0,2*ci+1]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[0,2*ci+1]=x[0,2*ci+1]*scalefac


  stride = patch_size//2 # patch stride (with 1/2 overlap)
  num_channels=8 # 4x2 polarizationx(real,imag)
  y = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
  # get new shape
  (nbase1,nchan1,patchx,patchy,nx,ny)=y.shape
  # create a new tensor
  y1=torch.zeros([nbase1*patchx*patchy,nchan1,nx,ny]).to(mydevice,non_blocking=True)

  # copy data ordered according to the patches
  ck=0
  for ci in range(patchx):
   for cj in range(patchy):
     y1[ck*nbase1:(ck+1)*nbase1,:,:,:]=y[:,:,ci,cj,:,:]
     ck=ck+1

  y = y1
  del x,y1
  # note: nbatch = batch_size x patchx x patchy
  #(nbatch,nchan,nxx,nyy)=y.shape

  # do some rough cleanup of data
  ##y[y!=y]=0 # set NaN,Inf to zero
  torch.clamp(y,-1e6,1e6) # clip high values

  # normalize data
  ymean=y.mean()
  ystd=y.std()
  y.sub_(ymean).div_(ystd)

  return patchx,patchy,y


########################################################
def get_metadata(filename,SAP):
  # open LOFAR H5 file, read metadata from a SAP,
  # return baselines, time, frequencies, polarizations, real/imag

  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']

  return g.shape
 
########################################################

########################################################
class AutoEncoderCNN(nn.Module):
    # AE CNN 
    def __init__(self,latent_dim=128,K=10,channels=3):
        super(AutoEncoderCNN,self).__init__()
        self.latent_dim=latent_dim
        self.K=K
        # 32x32 -> 16x16
        self.conv1=nn.Conv2d(channels, 12, 4, stride=2, padding=1)# in channels chan, out 12 chan, kernel 4x4
        # 16x16 -> 8x8
        self.conv2=nn.Conv2d(12, 24, 4, stride=2,  padding=1)# in 12 chan, out 24 chan, kernel 4x4
        # 8x8 -> 4x4
        self.conv3=nn.Conv2d(24, 48, 4, stride=2,  padding=1)# in 24 chan, out 48 chan, kernel 4x4
        # 4x4 -> 2x2
        self.conv4=nn.Conv2d(48, 96, 4, stride=2,  padding=1)# in 48 chan, out 96 chan, kernel 4x4

        self.fc1=nn.Linear(384,self.latent_dim)

        self.fc3=nn.Linear(self.latent_dim,384)
        self.tconv1=nn.ConvTranspose2d(96,48,4,stride=2,padding=1)
        self.tconv2=nn.ConvTranspose2d(48,24,4,stride=2,padding=1)
        self.tconv3=nn.ConvTranspose2d(24,12,4,stride=2,padding=1)
        self.tconv4=nn.ConvTranspose2d(12,channels,4,stride=2,padding=1)

    def forward(self, x):
        mu=self.encode(x)
        return self.decode(mu), mu

    def encode(self, x):
        #In  1,1,32,32
        x=F.elu(self.conv1(x)) # 1,12,16,16
        x=F.elu(self.conv2(x)) # 1,24,8,8
        x=F.elu(self.conv3(x)) # 1,48,4,4
        x=F.elu(self.conv4(x)) # 1,96,2,2
        x=torch.flatten(x,start_dim=1) # 1,96*2*2
        x=F.elu(self.fc1(x)) # 1,latent_dim
        return x # 1,latent_dim

    def decode(self, z):
        # In 1,latent_dim
        x=self.fc3(z) # 1,384
        x=torch.reshape(x,(-1,96,2,2)) # 1,96,2,2
        x=F.elu(self.tconv1(x)) # 1,48,4,4
        x=F.elu(self.tconv2(x)) # 1,24,8,8
        x=F.elu(self.tconv3(x)) # 1,12,16,16
        x=self.tconv4(x) # 1,8,32,32
        return torch.tanh(x) # 1,channels,32,32

########################################################
class AutoEncoderCNN1(nn.Module):
    # AE CNN 
    def __init__(self,latent_dim=128,K=10,channels=3):
        super(AutoEncoderCNN1,self).__init__()
        self.latent_dim=latent_dim
        self.K=K
        # 64x64 -> 32x32
        self.conv1=nn.Conv2d(channels, 12, 4, stride=2, padding=1)# in channels chan, out 12 chan, kernel 4x4
        # 32x32 -> 16x16
        self.conv2=nn.Conv2d(12, 24, 4, stride=2,  padding=1)# in 12 chan, out 24 chan, kernel 4x4
        # 16x16 -> 8x8
        self.conv3=nn.Conv2d(24, 48, 4, stride=2,  padding=1)# in 24 chan, out 48 chan, kernel 4x4
        # 8x8 -> 4x4
        self.conv4=nn.Conv2d(48, 96, 4, stride=2,  padding=1)# in 48 chan, out 96 chan, kernel 4x4
        # 4x4 -> 2x2
        self.conv5=nn.Conv2d(96, 192, 4, stride=2,  padding=1)# in 96 chan, out 192 chan, kernel 4x4
        # 2x2x192=768
        self.fc1=nn.Linear(768,self.latent_dim)

        self.fc3=nn.Linear(self.latent_dim,768)
        self.tconv0=nn.ConvTranspose2d(192,96,4,stride=2,padding=1)
        self.tconv1=nn.ConvTranspose2d(96,48,4,stride=2,padding=1)
        self.tconv2=nn.ConvTranspose2d(48,24,4,stride=2,padding=1)
        self.tconv3=nn.ConvTranspose2d(24,12,4,stride=2,padding=1)
        self.tconv4=nn.ConvTranspose2d(12,channels,4,stride=2,padding=1)

    def forward(self, x):
        mu=self.encode(x)
        return self.decode(mu), mu

    def encode(self, x):
        #In  1,1,64,64
        x=F.elu(self.conv1(x)) # 1,12,32,32
        x=F.elu(self.conv2(x)) # 1,24,16,16
        x=F.elu(self.conv3(x)) # 1,48,8,8
        x=F.elu(self.conv4(x)) # 1,96,4,4
        x=F.elu(self.conv5(x)) # 1,192,2,2
        x=torch.flatten(x,start_dim=1) # 1,192*2*2=768
        x=F.elu(self.fc1(x)) # 1,latent_dim
        return x # 1,latent_dim

    def decode(self, z):
        # In 1,latent_dim
        x=self.fc3(z) # 1,768
        x=torch.reshape(x,(-1,192,2,2)) # 1,192,2,2
        x=F.elu(self.tconv0(x)) # 1,96,4,4
        x=F.elu(self.tconv1(x)) # 1,48,8,8
        x=F.elu(self.tconv2(x)) # 1,24,16,16
        x=F.elu(self.tconv3(x)) # 1,12,32,32
        x=self.tconv4(x) # 1,channels,64,64
        return torch.tanh(x) # 1,channels,64,64


########################################################
#### K harmonic means module
class Kmeans(nn.Module):
  def __init__(self,latent_dim=128,K=10,p=2):
     super(Kmeans,self).__init__()
     self.latent_dim=latent_dim
     self.K=K
     self.p=p # K harmonic mean order 1/|| ||^p
     # cluster centroids
     self.M=torch.nn.Parameter(torch.rand(self.K,self.latent_dim),requires_grad=True)
  def forward(self,X):
     # calculate distance of each X from cluster centroids
     (nbatch,_)=X.shape
     loss=0
     for nb in range(nbatch):
       # calculate harmonic mean for x := K/ sum_k (1/||x-m_k||^p)
       ek=0
       for ck in range(self.K):
         ek=ek+1.0/torch.norm(self.M[ck,:]-X[nb,:],2)**(self.p)
       loss=loss+self.K/ek
     return loss/(nbatch*self.K)
########################################################
