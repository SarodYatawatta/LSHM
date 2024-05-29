import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

from lofar_models import *
from lofar_tools import *

from matplotlib import pyplot as plt

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import NNConv
import networkx as nx

use_cuda=False
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# Load pre-trained model to evaluate clustering for given LOFAR dataset

L=256-32 # latent dimension
Lt=16 # latent dimensions in time/frequency axes (1D CNN)
Kc=10 # K-harmonic clusters
Khp=4 # order of K harmonic mean 1/|| ||^p norm
Ko=10 # final hard clusters

# label size=distance to each cluster centroid: Kc 
n_label=Kc
# feature size=total latent dimension
n_features=L+2*Lt

patch_size=128

load_model=True

# enable this to create psuedocolor images using all XX and YY
colour_output=True

# reconstruction ICA
use_rica=True

num_in_channels=4 # real,imag XX,YY

# harmonic scales to use (sin,cos)(scale*u, scale*v) and so on
# these can be regarded as l,m sky coordinate distance where sources can be present
harmonic_scales=torch.tensor([1e-4, 1e-3, 1e-2, 1e-1]).to(mydevice)

# for 128x128 patches
net=AutoEncoderCNN2(latent_dim=L,channels=num_in_channels,harmonic_scales=harmonic_scales,rica=use_rica)

# 1D autoencoders
net1D1=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels,harmonic_scales=harmonic_scales,rica=use_rica)
net1D2=AutoEncoder1DCNN(latent_dim=Lt,channels=num_in_channels,harmonic_scales=harmonic_scales,rica=use_rica)
mod=Kmeans(latent_dim=(L+Lt+Lt),p=Khp)

if load_model:
  checkpoint=torch.load('./net.model',map_location=mydevice)
  net.load_state_dict(checkpoint['model_state_dict'])
  checkpoint=torch.load('./khm.model',map_location=mydevice)
  mod.load_state_dict(checkpoint['model_state_dict'])
  net.eval()
  mod.eval()

  checkpoint=torch.load('./netT.model',map_location=mydevice)
  net1D1.load_state_dict(checkpoint['model_state_dict'])
  checkpoint=torch.load('./netF.model',map_location=mydevice)
  net1D2.load_state_dict(checkpoint['model_state_dict'])
  net1D1.eval()
  net1D2.eval()

net.to(mydevice)
mod.to(mydevice)
net1D1.to(mydevice)
net1D2.to(mydevice)


file_list,sap_list=get_fileSAP('/home/sarod/scratch1/H5')

# dictionary to map station name to number
stations=dict()
station_id=0
# dictionary to map station1,station2 to baseline id
baseline_map=dict()
baseline_id=0

# go through all SAPs and collect all unique stations
for which_sap in range(len(file_list)):
    baselines,_=get_metadata(file_list[which_sap],sap_list[which_sap],give_baseline=True)
    for baseline in baselines:
        if baseline[0] not in stations.keys():
          stations[baseline[0]]=station_id
          station_id=station_id+1
        if baseline[1] not in stations.keys():
          stations[baseline[1]]=station_id
          station_id=station_id+1
        # exclude autocorrelations for baslines
        if baseline[0] != baseline[1]:
          if (baseline[0],baseline[1]) not in baseline_map.keys():
            baseline_map[(baseline[0],baseline[1])]=baseline_id
            baseline_id=baseline_id+1
          if (baseline[1],baseline[0]) not in baseline_map.keys():
            baseline_map[(baseline[1],baseline[0])]=baseline_id
            baseline_id=baseline_id+1

    print(f'SAP {which_sap} baselines {len(baselines)}')

n_stat=len(stations.keys())
n_base=len(baseline_map.keys())
print(f'total stations {n_stat} total baselines {n_base}')

# Geometry of the full graph:
# nodes: n_nodes=n_stat
# edges: n_edges=n_base=2*(n_stat-1)*n_stat/2 (both 0->1 and 1->0 edges need to be considered)
# data.x: node features (using autocorrelations) n_nodes x n_node_features
# data.edge_index : all edge info (each baseline included twice) 2 x n_edges
# data.edge_attr : edge features (using cross correlations) n_edges x n_edge_features
# data.y : target, node level, size n_nodes x n_labels or ...
# data.train_mask: masking array for edges 1 x n_edges


# data.edge_index
edge_index=torch.zeros((2,2*n_base),dtype=torch.long).contiguous().to(mydevice)
# data.edge_attr, edge attribute: latent variable for each input
edge_attr=torch.zeros((2*n_base,n_features),dtype=torch.float).contiguous().to(mydevice)
# data.train_mask, edge mask, all set to False (all excluded)
edge_mask=torch.zeros(2*n_base,dtype=torch.bool).contiguous().to(mydevice)

# data.x node attribute: each node will have autoencoder latent dim
node_attr=torch.zeros((n_stat,n_features),dtype=torch.float).contiguous().to(mydevice)
# data.y node labells: Kc
node_labels=torch.zeros((n_stat,n_label),dtype=torch.float).contiguous().to(mydevice)
# mask for nodes
node_mask=torch.zeros(n_stat,dtype=torch.bool).contiguous().to(mydevice)

class GraphNet(nn.Module):
    def __init__(self,edge_features=n_features,node_features=n_features,out_labels=3):
        super(GraphNet,self).__init__()
        # MLP to map edge features to node features
        nn1=nn.Sequential(nn.Linear(edge_features,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,node_features*out_labels))
        self.conv1=NNConv(node_features,out_labels,nn1,aggr='mean')

    def forward(self,data):
        xx=data.x[data.node_mask]
        edges=data.edge_index[:,data.edge_mask]
        edge_attr=data.edge_attr[data.edge_mask]
        x=self.conv1(xx,edges,edge_attr)
        return torch.softmax(x,0)


gnet=GraphNet(edge_features=n_features,node_features=n_features,out_labels=n_label)
gnet.train()
optimizer=torch.optim.Adam(gnet.parameters(),lr=0.01)
criterion=torch.nn.MSELoss()


for nepoch in range(20):
   which_sap=np.random.choice(len(file_list)) # valid in file_list/sap_list

   # get nbase,nfreq,ntime,npol,ncomplex
   baselines,(nbase,nfreq,ntime,npol,ncomplex)=get_metadata(file_list[which_sap],sap_list[which_sap],give_baseline=True)

   # iterate over each baselines
   with torch.no_grad():
     for nb in range(nbase):#range(nbase):
       baseline,patchx,patchy,x,uvcoords=get_data_for_baseline(file_list[which_sap],sap_list[which_sap],baseline_id=nb,patch_size=128,num_channels=num_in_channels,give_baseline=True,uvdist=True)

       # catch stations not in dict
       assert((baseline[0] in stations.keys()) and (baseline[1] in stations.keys()))

       is_autocorr=False

       # auto-correlations
       if baseline[0]==baseline[1]:
          is_autocorr=True
          station_id=stations[baseline[0]]
          node_mask[station_id]=True

       if not is_autocorr:
          edge_id=baseline_map[(baseline[0],baseline[1])]
          # add to edges
          edge_index[0,edge_id]=stations[baseline[0]]
          edge_index[1,edge_id]=stations[baseline[1]]
          edge_mask[edge_id]=True

       # x: have more than one (a batch), randomly select one item
       (nbatch,_,_,_)=x.shape
       nsel=np.random.choice(nbatch)
       x=x[nsel].to(mydevice)
       x=x.unsqueeze(0)
       uv=Variable(uvcoords[nsel].unsqueeze(0)).to(mydevice)
       # get latent variable
       x1,mu=net(x,uv)
       x11=(x-x1)/2
       # vectorize
       iy1=torch.flatten(x11,start_dim=2,end_dim=3)
       iy2=torch.flatten(torch.transpose(x11,2,3),start_dim=2,end_dim=3)
       yy1,yy1mu=net1D1(iy1,uv)
       yy2,yy2mu=net1D2(iy2,uv)
       # latent embedding
       Mu=torch.cat((mu,yy1mu,yy2mu),1)
       dist=torch.zeros(Kc)
       for ck in range(Kc):
          dist[ck]=torch.sum(torch.pow(torch.linalg.norm(Mu[0,:]-mod.M[ck,:],2),Khp))
       # map to probabilities (smallest dist ~ highest prob)
       dist=torch.softmax(-dist/dist.mean(),0)
       if is_autocorr:
          station_id=stations[baseline[0]]
          node_labels[station_id]=dist.flatten()
          node_attr[station_id]=Mu
       else:
          edge_id=baseline_map[(baseline[0],baseline[1])]
          edge_attr[edge_id,:]=Mu
          edge_mask[edge_id]=True

       if not is_autocorr:
          edge_id = baseline_map[(baseline[1],baseline[0])]
          # conjugate x (for baseline stat2 -> stat1 edge)
          # so multiply channels 1:2:4 by -1
          x[:,1,:] *=-1
          x[:,3,:] *=-1
          x1,mu=net(x,uv)
          x11=(x-x1)/2
          # vectorize
          iy1=torch.flatten(x11,start_dim=2,end_dim=3)
          iy2=torch.flatten(torch.transpose(x11,2,3),start_dim=2,end_dim=3)
          yy1,yy1mu=net1D1(iy1,uv)
          yy2,yy2mu=net1D2(iy2,uv)
          Mu=torch.cat((mu,yy1mu,yy2mu),1)
          dist=torch.zeros(Kc)
          for ck in range(Kc):
            dist[ck]=torch.sum(torch.pow(torch.linalg.norm(Mu[0,:]-mod.M[ck,:],2),Khp))
          dist=torch.softmax(-dist/dist.mean(),0)
          edge_attr[edge_id,:]=Mu
          edge_mask[edge_id]=True
          # add to edges
          edge_index[1,edge_id]=stations[baseline[0]]
          edge_index[0,edge_id]=stations[baseline[1]]
 

   #print('edge index')
   #print(edge_index[:,edge_mask])
   #print('edge mask')
   #print(edge_mask)
   #print('edge attr')
   #print(edge_attr[edge_mask])
   #print('node attr')
   #print(node_attr[node_mask])
   #print('node labels')
   #print(node_labels[node_mask])
   #print('node mask')
   #print(node_mask)

   graphdata=Data(x=node_attr,edge_index=edge_index,edge_attr=edge_attr,edge_mask=edge_mask,y=node_labels,node_mask=node_mask)

   print(f'node data {node_attr.shape} edge attr {edge_attr.shape} edge_index {edge_index.shape}')

   for n_iter in range(20):
      optimizer.zero_grad()
      gx=gnet(graphdata)
      loss=criterion(gx,graphdata.y[graphdata.node_mask])
      loss.backward()
      print(f'iter {n_iter} {loss.data.item()}')
      optimizer.step()
