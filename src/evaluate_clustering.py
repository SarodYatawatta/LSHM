from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py

import torch.fft

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

from sklearn.cluster import AgglomerativeClustering

# Load pre-trained model to evaluate clustering for given LOFAR dataset

L=256 # latent dimension
Lf=64 # latent dimension
Kc=10 # clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm

patch_size=128

# enable this to create psuedocolor images using all XX and YY
colour_output=True

from lofar_models import *

num_in_channels=4 # real,imag XX,YY
# 32x32 patches
#net=AutoEncoderCNN(latent_dim=L,K=Kc,channels=8)
# 64x64 patches
#net=AutoEncoderCNN1(latent_dim=L,K=Kc,channels=8)
# for 128x128 patches
net=AutoEncoderCNN2(latent_dim=L,channels=num_in_channels)
# fft: real,imag, so increase number of channels
fnet=AutoEncoderCNN2(latent_dim=Lf,channels=2*num_in_channels)
mod=Kmeans(latent_dim=(L+Lf),p=Khp)

checkpoint=torch.load('./net.model',map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['model_state_dict'])
checkpoint=torch.load('./fnet.model',map_location=mydevice)
fnet.load_state_dict(checkpoint['model_state_dict'])
checkpoint=torch.load('./khm.model',map_location=torch.device('cpu'))
mod.load_state_dict(checkpoint['model_state_dict'])

torchvision.utils.save_image(mod.M.data,'M.png')
file_list=['/home/sarod/L785751.MS_extract.h5','/home/sarod/L785751.MS_extract.h5',
   '/home/sarod/L785747.MS_extract.h5', '/home/sarod/L785757.MS_extract.h5',
   '/home/sarod/L696315.MS_extract.h5', '/home/sarod/L696315.MS_extract.h5',
   '/home/sarod/L686974.MS_extract.h5', '/home/sarod/L686974.MS_extract.h5',
   '/home/sarod/L798736.MS_extract.h5', '/home/sarod/L775633.MS_extract.h5',
   '/home/sarod/L684188.MS_extract.h5', '/home/sarod/L672470.MS_extract.h5',
   '/home/sarod/L672470.MS_extract.h5'
  ]
sap_list=['1','2','0','0','1','2','1','2','0','0','1','1','2']

which_sap=2 # valid in file_list/sap_list

# get nbase,nfreq,ntime,npol,ncomplex
nbase,nfreq,ntime,npol,ncomplex=get_metadata(file_list[which_sap],sap_list[which_sap])

X=np.zeros([Kc,nbase],dtype=np.float)
clusid=np.zeros(nbase,dtype=np.float)


# iterate over each baselines
for nb in range(nbase):
 patchx,patchy,x=get_data_for_baseline(file_list[which_sap],sap_list[which_sap],baseline_id=nb,patch_size=128,num_channels=num_in_channels)
 # get latent variable
 xhat,mu=net(x)
 # perform 2D fft
 fftx=torch.fft.fftn(x-xhat,dim=(2,3),norm='ortho')
 # fftshift
 freal,fimag=torch_fftshift(fftx.real,fftx.imag)
 y=torch.cat((freal,fimag),1)
 # clamp high values data
 y.clamp_(min=-10,max=10)
 yhat,ymu=fnet(y)
 if not colour_output:
  torchvision.utils.save_image( torch.cat((torch.cat((x[0,1],xhat[0,1])),(patch_size*patch_size)*torch.cat((y[0,1],yhat[0,1]))),1).data, 'xx_'+str(nb)+'.png' )
 else:
  x0=channel_to_rgb(x[0])
  xhat0=channel_to_rgb(xhat[0])
  y0=channel_to_rgb(y[0,0:4])
  yhat0=channel_to_rgb(yhat[0,0:4])
  torchvision.utils.save_image( torch.cat((torch.cat((x0,xhat0),1),torch.cat((y0,yhat0),1)),2).data, 'xx_'+str(nb)+'.png' )
 Mu=torch.cat((mu,ymu),1)
 kdist=mod(Mu)
 (nbatch,_)=Mu.shape
 dist=torch.zeros(Kc)
 for ck in range(Kc):
   for cn in range(nbatch):
     dist[ck]=dist[ck]+(torch.norm(Mu[cn,:]-mod.M[ck,:],2))
 dist=dist/nbatch
 X[:,nb]=dist.detach().numpy()
 (values,indices)=torch.min(dist.view(Kc,1),0)
 print('%d %e %d'%(nb,kdist,indices[0])) 
 clusid[nb]=indices[0]




### tSNE
tsne=TSNE(verbose=True)
X_emb=tsne.fit_transform(X.transpose())
uniq=np.unique(clusid)
snsplot=sns.scatterplot(X_emb[:,0], X_emb[:,1], hue=clusid, legend='full', 
  palette = sns.color_palette("bright", n_colors=len(uniq)))
snsplot.figure.savefig('scatter.png')

### final clustering
db=AgglomerativeClustering(linkage='average',n_clusters=10).fit(X_emb)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise_ = list(db.labels_).count(-1)

# Black removed and is used for noise instead.
unique_labels = set(db.labels_)
colors = [plt.cm.Spectral(each)
  for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
  class_member_mask = (db.labels_ == k)
  xy = X_emb[class_member_mask]
  plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    markeredgecolor='k', markersize=14)

plt.legend(labels=np.unique(db.labels_))
plt.title('Number of clusters: %d' % n_clusters_)
plt.savefig('clusters.png')


for nb in range(nbase):
 vis=get_data_for_baseline_flat(file_list[which_sap],sap_list[which_sap],baseline_id=nb,num_channels=num_in_channels)
 if not colour_output:
  torchvision.utils.save_image(vis[0,0].data, 'b'+str(db.labels_[nb])+'_'+str(nb)+'.png')
 else:
  torchvision.utils.save_image(channel_to_rgb(vis[0]), 'b'+str(db.labels_[nb])+'_'+str(nb)+'.png')
