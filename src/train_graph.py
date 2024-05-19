import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py

from lofar_tools import *
from lofar_models import *

from matplotlib import pyplot as plt

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
import networkx as nx

use_cuda=True
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

which_sap=0 # valid in file_list/sap_list -7

# get nbase,nfreq,ntime,npol,ncomplex
baselines,(nbase,nfreq,ntime,npol,ncomplex)=get_metadata(file_list[which_sap],sap_list[which_sap],give_baseline=True)

# for debugging use a low value
#NBASE=1000
NBASE=nbase

#########################################################################
def build_edge_graph(baselines):
 # build edge graph (returns edge index, indexed according to basline number)
 # nodes are the baselines
 # for basline sta1,sta2 : edges for all other balines having sta1 or sta2
  (nbase,_)=baselines.shape
  # station to basline mapping
  stations=dict()
  for nb in range(NBASE):#range(nbase):
    bline=baselines[nb]
    if bline[0] not in stations.keys():
     stations[bline[0]]=list()
    if nb not in stations[bline[0]]:
      stations[bline[0]].append(nb)
    if bline[1] not in stations.keys():
     stations[bline[1]]=list()
    if nb not in stations[bline[1]]:
     stations[bline[1]].append(nb)
  #print(stations)
  edgelist=list()
  for nb in range(NBASE):#range(nbase):
    bline=baselines[nb]
    for id1 in stations[bline[0]]:
      edgelist.append([nb, id1])
    if bline[0] != bline[1]:
      for id1 in stations[bline[1]]:
        if id1 !=nb:
         edgelist.append([nb, id1])
 
  edge_index=np.array(edgelist) 
  return edge_index
#########################################################################
edge_index_np=build_edge_graph(baselines)
# edge index: baseline id
edge_index=torch.tensor(edge_index_np.transpose(),dtype=torch.long).contiguous().to(mydevice)

# get one baseline to get patch sizes
patchx,patchy,x=get_data_for_baseline(file_list[which_sap],sap_list[which_sap],baseline_id=0,patch_size=128,num_channels=num_in_channels)

# feature size = latent size
Nfeat=L+Lt+Lt
# edge attribute: None
edge_attr=None
# node attribute: latent features of each baseline
node_data=torch.zeros((NBASE,Nfeat)).to(mydevice)
# node label: distance from Kharmonic means : Kc values
node_label=torch.zeros((NBASE,Kc)).to(mydevice)

# iterate over each baselines
with torch.no_grad():
 for nb in range(NBASE):#range(nbase):
   baseline,patchx,patchy,x,uvcoords=get_data_for_baseline(file_list[which_sap],sap_list[which_sap],baseline_id=nb,patch_size=128,num_channels=num_in_channels,give_baseline=True,uvdist=True)
   x=x.to(mydevice)
   uv=Variable(uvcoords).to(mydevice)
   # get latent variable
   x1,mu=net(x,uv)
   x11=(x-x1)/2
   # vectorize
   iy1=torch.flatten(x11,start_dim=2,end_dim=3)
   iy2=torch.flatten(torch.transpose(x11,2,3),start_dim=2,end_dim=3)
   yy1,yy1mu=net1D1(iy1,uv)
   yy2,yy2mu=net1D2(iy2,uv)
   Mu=torch.cat((mu,yy1mu,yy2mu),1)
   # average over batch dimension : patchx x patchy
   (nbatch,_)=Mu.shape
   dist=torch.zeros(Kc).to(mydevice)
   for ck in range(Kc):
     for cn in range(nbatch):
       dist[ck]=dist[ck]+torch.sum(torch.linalg.norm(Mu[cn,:]-mod.M[ck,:],2))
   dist=dist/nbatch
   node_data[nb,:]=torch.mean(Mu,dim=0)
   node_label[nb,:]=dist



graphdata=Data(x=node_data,edge_index=edge_index,y=node_label)

print(f'Graph has {graphdata.num_nodes} nodes and {graphdata.num_edges} edges')
print(f'has self loops {graphdata.has_self_loops()} is undirected {graphdata.is_undirected()}')

def visualize(h,color=None,epoch=None,loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h=h.detach().cpu().numpy()
        plt.scatter(h[:,0],h[:,1],s=140,c=color,cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}',fontsize=16)
    else:
        nx.draw_networkx(G,pos=nx.spring_layout(G,seed=42), with_labels=True,
                    node_color=color, cmap="Set2")
    plt.show()


G=to_networkx(graphdata,to_undirected=True)
visualize(G)

class GraphNet(nn.Module):
    def __init__(self,node_features=Nfeat,node_labels=Kc,hidden_channels=4):
        super(GraphNet,self).__init__()
        self.conv1=GCNConv(node_features,hidden_channels)
        self.conv2=GCNConv(hidden_channels,node_labels)
    def forward(self,x,edge_index):
        x=self.conv1(x,edge_index)
        x=x.relu()
        x=self.conv2(x,edge_index)
        return x


gnet=GraphNet(node_features=Nfeat,node_labels=Kc,hidden_channels=4).to(mydevice)
gnet.train()
optimizer=torch.optim.Adam(gnet.parameters(),lr=0.01)
criterion=torch.nn.MSELoss()
for nepoch in range(200):
  optimizer.zero_grad()
  gx=gnet(graphdata.x,graphdata.edge_index)
  loss=criterion(gx,graphdata.y)
  loss.backward()
  print(loss.data.item())
  optimizer.step()

