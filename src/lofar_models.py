from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

# This file contains various models used

########################################################
class AutoEncoderCNN2(nn.Module):
    # AE CNN 
    def __init__(self,latent_dim=128,channels=3,harmonic_scales=None):
        super(AutoEncoderCNN2,self).__init__()
        self.latent_dim=latent_dim
        # scale factors for harmonics of u,v coords
        self.harmonic_scales=harmonic_scales
        # harmonic dim: H x 2(u,v) x 2(cos,sin), H from above
        self.harmonic_dim=(self.harmonic_scales.size()[0])*2*2
        # 128x128 -> 64x64
        self.conv0=nn.Conv2d(channels, 8, 4, stride=2, padding=1)# in channels chan, out 8 chan, kernel 4x4
        # 64x64 -> 32x32
        self.conv1=nn.Conv2d(8, 12, 4, stride=2, padding=1)# in channels 8, out 12 chan, kernel 4x4
        # 32x32 -> 16x16
        self.conv2=nn.Conv2d(12, 24, 4, stride=2,  padding=1)# in 12 chan, out 24 chan, kernel 4x4
        # 16x16 -> 8x8
        self.conv3=nn.Conv2d(24, 48, 4, stride=2,  padding=1)# in 24 chan, out 48 chan, kernel 4x4
        # 8x8 -> 4x4
        self.conv4=nn.Conv2d(48, 96, 4, stride=2,  padding=1)# in 48 chan, out 96 chan, kernel 4x4
        # 4x4 -> 2x2
        self.conv5=nn.Conv2d(96, 192, 4, stride=2,  padding=1)# in 96 chan, out 192 chan, kernel 4x4
        # Linear layers to operate on u,v coordinate harmonics
        self.fcuv1=nn.Linear(self.harmonic_dim,self.harmonic_dim)
        self.fcuv3=nn.Linear(self.harmonic_dim,self.harmonic_dim)
        # 2x2x192=768
        self.fc1=nn.Linear(768+self.harmonic_dim,self.latent_dim)

        self.fc3=nn.Linear(self.latent_dim+self.harmonic_dim,768)
        self.tconv0=nn.ConvTranspose2d(192,96,4,stride=2,padding=1)
        self.tconv1=nn.ConvTranspose2d(96,48,4,stride=2,padding=1)
        self.tconv2=nn.ConvTranspose2d(48,24,4,stride=2,padding=1)
        self.tconv3=nn.ConvTranspose2d(24,12,4,stride=2,padding=1)
        self.tconv4=nn.ConvTranspose2d(12,8,4,stride=2,padding=1)
        self.tconv5=nn.ConvTranspose2d(8,channels,4,stride=2,padding=1)

    def forward(self,x,uv):
        uv=torch.kron(self.harmonic_scales,uv)
        uv=torch.cat((torch.sin(uv),torch.cos(uv)),dim=1)
        uv=torch.flatten(uv,start_dim=1)
        mu=self.encode(x,uv)
        return self.decode(mu,uv),mu

    def encode(self,x,uv):
        #In  1,4,128,128
        x=F.elu(self.conv0(x)) # 1,8,64,64
        x=F.elu(self.conv1(x)) # 1,12,32,32
        x=F.elu(self.conv2(x)) # 1,24,16,16
        x=F.elu(self.conv3(x)) # 1,48,8,8
        x=F.elu(self.conv4(x)) # 1,96,4,4
        x=F.elu(self.conv5(x)) # 1,192,2,2
        x=torch.flatten(x,start_dim=1) # 1,192*2*2=768
        uv=F.elu(self.fcuv1(uv))
        # combine uv harmonics
        x=torch.cat((x,uv),dim=1)
        x=F.elu(self.fc1(x)) # 1,latent_dim
        return x # 1,latent_dim

    def decode(self,z,uv):
        # In z: 1,latent_dim
        # harmonic input
        uv=F.elu(self.fcuv3(uv))
        z=torch.cat((z,uv),dim=1)
        x=self.fc3(z) # 1,768
        x=torch.reshape(x,(-1,192,2,2)) # 1,192,2,2
        x=F.elu(self.tconv0(x)) # 1,96,4,4
        x=F.elu(self.tconv1(x)) # 1,48,8,8
        x=F.elu(self.tconv2(x)) # 1,24,16,16
        x=F.elu(self.tconv3(x)) # 1,12,32,32
        x=F.elu(self.tconv4(x)) # 1,8,64,64
        x=self.tconv5(x) # 1,channels,128,128
        return x # 1,channels,128,128


########################################################
class AutoEncoder1DCNN(nn.Module):
    # 1 dimensional AE CNN 
    def __init__(self,latent_dim=128,channels=3):
        super(AutoEncoder1DCNN,self).__init__()
        self.latent_dim=latent_dim
        # all dimensions below are vectorized values
        # 128^2x 1  -> 64^2x 1
        self.conv0=nn.Conv1d(channels, 8, 4, stride=4, padding=1)# in channels chan, out 8 chan, kernel 4x4
        # 64^2x1 -> 32^2x1
        self.conv1=nn.Conv1d(8, 12, 4, stride=4, padding=1)# in channels 8, out 12 chan, kernel 4x4
        # 32^2x1 -> 16^2x1
        self.conv2=nn.Conv1d(12, 24, 4, stride=4,  padding=1)# in 12 chan, out 24 chan, kernel 4x4
        # 16^2x1 -> 8^2x1
        self.conv3=nn.Conv1d(24, 48, 4, stride=4,  padding=1)# in 24 chan, out 48 chan, kernel 4x4
        # 8^2x1 -> 4^2x1
        self.conv4=nn.Conv1d(48, 96, 4, stride=4,  padding=1)# in 48 chan, out 96 chan, kernel 4x4
        # 4^2x1 -> 2^2x1
        self.conv5=nn.Conv1d(96, 192, 4, stride=4,  padding=1)# in 96 chan, out 192 chan, kernel 4x4
        # 2^2x192=768
        self.fc1=nn.Linear(768,self.latent_dim)

        self.fc3=nn.Linear(self.latent_dim,768)
        # output_padding is added to match the input sizes
        self.tconv0=nn.ConvTranspose1d(192,96,4,stride=4,padding=0,output_padding=0)
        self.tconv1=nn.ConvTranspose1d(96,48,4,stride=4,padding=0,output_padding=0)
        self.tconv2=nn.ConvTranspose1d(48,24,4,stride=4,padding=0,output_padding=0)
        self.tconv3=nn.ConvTranspose1d(24,12,4,stride=4,padding=0,output_padding=0)
        self.tconv4=nn.ConvTranspose1d(12,8,4,stride=4,padding=0,output_padding=0)
        self.tconv5=nn.ConvTranspose1d(8,channels,4,stride=4,padding=0,output_padding=0)

    def forward(self, x):
        mu=self.encode(x)
        return self.decode(mu), mu

    def encode(self, x):
        #In  1,4,128^2
        x=F.elu(self.conv0(x)) # 1,8,64^2
        x=F.elu(self.conv1(x)) # 1,12,32^2
        x=F.elu(self.conv2(x)) # 1,24,16^2
        x=F.elu(self.conv3(x)) # 1,48,8^2
        x=F.elu(self.conv4(x)) # 1,96,4^2
        x=F.elu(self.conv5(x)) # 1,192,2^2
        x=torch.flatten(x,start_dim=1) # 1,192*2*2=768
        x=F.elu(self.fc1(x)) # 1,latent_dim
        return x # 1,latent_dim

    def decode(self, z):
        # In 1,latent_dim
        x=self.fc3(z) # 1,768
        x=torch.reshape(x,(-1,192,2*2)) # 1,192,2^2
        x=F.elu(self.tconv0(x)) # 1,96,4^2
        x=F.elu(self.tconv1(x)) # 1,48,8^2
        x=F.elu(self.tconv2(x)) # 1,24,16^2
        x=F.elu(self.tconv3(x)) # 1,12,32^2
        x=F.elu(self.tconv4(x)) # 1,8,64^2
        x=self.tconv5(x) # 1,channels,128^2
        return x # 1,channels,128^2


########################################################
#### K harmonic means module
class Kmeans(nn.Module):
  def __init__(self,latent_dim=128,K=10,p=2):
     super(Kmeans,self).__init__()
     self.latent_dim=latent_dim
     self.K=K
     self.p=p # K harmonic mean order 1/|| ||^p
     self.EPS=1e-9# epsilon to avoid 1/0 cases
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
         ek=ek+1.0/(torch.pow(torch.linalg.norm(self.M[ck,:]-X[nb,:],2),self.p)+self.EPS)
       loss=loss+self.K/(ek+self.EPS)
     return loss/(nbatch*self.K*self.latent_dim)

  def clustering_error(self,X):
    return self.forward(X)

  def cluster_similarity(self):
     # use contrastive loss variant
     # for each row k, denominator=exp(zk^T zk/||zk||^2)
     # numerator = sum_l,l\ne k exp(zk^T zl / ||zk|| ||zl||)
     loss=0
     # take outer product between each rows
     for ci in range(self.K):
       mnrm=torch.linalg.norm(self.M[ci,:],2)
       # denominator is actually=1
       denominator=torch.exp(torch.dot(self.M[ci,:],self.M[ci,:])/(mnrm*mnrm+self.EPS))
       numerator=0
       for cj in range(self.K):
        if cj!=ci:
          numerator=numerator+torch.exp(torch.dot(self.M[ci,:],self.M[cj,:])/(mnrm*torch.linalg.norm(self.M[cj,:],2)+self.EPS))
       loss=loss+(numerator/(denominator+self.EPS))
     return loss/(self.K*self.latent_dim)

  def offline_update(self,X):
      # update cluster centroids using recursive formula
      # Eq (7.1-7.5) of B. Zhang - generalized K-harmonic means
      (nbatch,_)=X.shape
      alpha=torch.zeros(nbatch)
      Q=torch.zeros(nbatch,self.K)
      q=torch.zeros(self.K)
      P=torch.zeros(nbatch,self.K)
      # indices i=1..nbatch, k or j=1..K
      for ci in range(nbatch):
        # alpha_i := 1/ (sum_k (1/||x_i-m_k||^p))^2
        ek=0
        for ck in range(self.K):
          ek=ek+1.0/(torch.pow(torch.linalg.norm(self.M[ck,:]-X[ci,:],2),self.p)+self.EPS)
        alpha[ci]=1.0/(ek**2+self.EPS)
        # Q_ij = alpha_i/ ||x_i-m_j||^(p+2)
        for ck in range(self.K):
          Q[ci,ck]=alpha[ci]/(torch.pow(torch.linlag.norm(self.M[ck,:]-X[ci,:],2),self.p+2)+self.EPS)
      # q_j = sum_i Q_ij
      for ck in range(self.K):
          q[ck]=torch.sum(Q[:,ck])
      # P_ij = Q_ij/q_j
      for ci in range(nbatch):
        for ck in range(self.K):
          P[ci,ck]=Q[ci,ck]/q[ck]
      # M_j = sum_i P_ij x_i
      for ck in range(self.K):
        self.M[ck,:]=0
        for ci in range(nbatch):
          self.M[ck,:]+=P[ci,ck]*X[ci,:]
      del P,Q,q,alpha
########################################################

