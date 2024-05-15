import torch
import torchvision
import numpy as np
import h5py

import torch.fft

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load pre-trained model to evaluate clustering for given LOFAR dataset

L=256-32#256 # latent dimension
Lt=16#32 # latent dimensions in time/frequency axes (1D CNN)
Kc=10 # K-harmonic clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm
Ko=10 # final hard clusters
# reconstruction ICA
use_rica=True

patch_size=128

# enable this to create psuedocolor images using all XX and YY
colour_output=True

from lofar_tools import *
from lofar_models import *

num_in_channels=4 # real,imag XX,YY

# harmonic scales to use (sin,cos)(scale*u, scale*v) and so on
harmonic_scales=torch.tensor([1e-4, 1e-3, 1e-2, 1e-1]).to('cpu')

# for 128x128 patches
net=AutoEncoderCNN2(latent_dim=L,channels=num_in_channels,harmonic_scales=harmonic_scales,rica=use_rica)

# 1D autoencoders
net1D1=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels,harmonic_scales=harmonic_scales,rica=use_rica)
net1D2=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels,harmonic_scales=harmonic_scales,rica=use_rica)
mod=Kmeans(latent_dim=(L+Lt+Lt),K=Kc,p=Khp)

checkpoint=torch.load('./net.model',map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['model_state_dict'])
checkpoint=torch.load('./khm.model',map_location=torch.device('cpu'))
mod.load_state_dict(checkpoint['model_state_dict'])
net.eval()
mod.eval()

checkpoint=torch.load('./netT.model',map_location=torch.device('cpu'))
net1D1.load_state_dict(checkpoint['model_state_dict'])
checkpoint=torch.load('./netF.model',map_location=torch.device('cpu'))
net1D2.load_state_dict(checkpoint['model_state_dict'])
net1D1.eval()
net1D2.eval()

torchvision.utils.save_image(mod.M.data,'M.png')
mydict={'M':mod.M.data.numpy()}
from scipy.io import savemat
savemat('M.mat',mydict)
file_list,sap_list=get_fileSAP('/media/sarod')
which_sap=-16 # valid in file_list/sap_list -7

# get nbase,nfreq,ntime,npol,ncomplex
nbase,nfreq,ntime,npol,ncomplex=get_metadata(file_list[which_sap],sap_list[which_sap])

X=np.zeros([Kc,nbase],dtype=np.float64)
clusid=np.zeros(nbase,dtype=np.float64)

# iterate over each baselines
with torch.no_grad():
  for nb in range(nbase):
   patchx,patchy,x,uvcoords=get_data_for_baseline(file_list[which_sap],sap_list[which_sap],baseline_id=nb,patch_size=128,num_channels=num_in_channels,uvdist=True)
   x=x.cpu() # send to cpu
   uv=uvcoords.cpu()
   # get latent variable
   x1,mu=net(x,uv)
   x11=(x-x1)/2
   # vectorize
   iy1=torch.flatten(x11,start_dim=2,end_dim=3)
   iy2=torch.flatten(torch.transpose(x11,2,3),start_dim=2,end_dim=3)
   yy1,yy1mu=net1D1(iy1,uv)
   yy2,yy2mu=net1D2(iy2,uv)
   x2=yy1.view_as(x)
   x3=torch.transpose(yy2.view_as(x),2,3)
   # reconstruction
   xrecon=x1+x2+x3
   if not colour_output:
    torchvision.utils.save_image( torch.cat((torch.cat((x[0,1],x1[0,1])),
     torch.cat((x2[0,1],x3[0,1]))
     ),1).data, 'xx_'+str(nb)+'.png' )
   else:
    x0=channel_to_rgb(x[0])
    xhat0=channel_to_rgb(x1[0])
    y1D1=channel_to_rgb(x2[0,0:4])
    y1D2=channel_to_rgb(x3[0,0:4])
    xrec=channel_to_rgb(xrecon[0,0:4])
    xerr=channel_to_rgb(x[0,0:4]-xrecon[0,0:4])
    print("norm x=%f xhat=%f"%(torch.linalg.norm(x0),
       torch.linalg.norm(xhat0)))
    torchvision.utils.save_image( torch.cat((torch.cat((x0,xhat0),1),
       torch.cat((y1D1,y1D2),1),torch.cat((xrec,xerr),1)),
       2).data, 'xx_'+str(nb)+'.png' )
   Mu=torch.cat((mu,yy1mu,yy2mu),1)
   kdist=mod(Mu)
   (nbatch,_)=Mu.shape
   dist=torch.zeros(Kc)
   for ck in range(Kc):
     for cn in range(nbatch):
       dist[ck]=dist[ck]+torch.sum(torch.pow(torch.linalg.norm(Mu[cn,:]-mod.M[ck,:],2),Khp))
   dist=dist/nbatch 
   X[:,nb]=dist.detach().numpy()
   (values,indices)=torch.min(dist.view(Kc,1),0)
   print('%d %e %d'%(nb,kdist,indices[0])) 
   clusid[nb]=indices[0]

# subtract mean from each row of X
for ck in range(Kc):
 X[ck]=X[ck]-np.mean(X[ck])

mydict={'X':X}
savemat('X.mat',mydict)

### tSNE
tsne=TSNE(n_components=2,random_state=99,verbose=True)
X_emb=tsne.fit_transform(X.transpose())
uniq=np.unique(clusid)
snsplot=sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=clusid, legend='full',
  palette = sns.color_palette("bright", n_colors=len(uniq)))
snsplot.figure.savefig('scatter.png')

### final clustering
scaler=StandardScaler()
X_embsc=scaler.fit_transform(X_emb)
db=AgglomerativeClustering(linkage='average',n_clusters=Ko).fit(X_embsc)
#db=DBSCAN(eps=0.3).fit(X_embsc)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

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
