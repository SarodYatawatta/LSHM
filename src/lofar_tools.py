from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py
import glob
import os,math

# This file contains various routines used in reading LOFAR H5 data

# Should we recursively search for training data?
rec_file_search=True
# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

########################################################
def torch_fftshift(real, imag):
  # FFTshift method, since torch does not have it yet
  # only work with dims 2,3
  for dim in range(2, len(real.size())):
    real = torch.roll(real, dims=dim, shifts=real.size(dim)//2)
    imag = torch.roll(imag, dims=dim, shifts=imag.size(dim)//2)
  return real, imag

###############################
def channel_to_rgb(x):
  # x: 4 x nx x ny image
  # output 3 x nx ny image for RGB plot
  (nchan,nx,ny)=x.shape
  assert(nchan==4)

  # normalize
  xmean=x.mean()
  xstd=x.std()
  x.sub_(xmean).div_(xstd)

  y=torch.zeros(3,nx,ny)
  y[0]=(x[0]+0.3*x[1])/1.3
  y[1]=(0.7*x[1]+0.7*x[2])/1.4
  y[2]=(0.3*x[2]+x[3])/1.3
  return y

########################################################
def get_data_minibatch(file_list,SAP_list,batch_size=2,patch_size=32,normalize_data=False,num_channels=8,transform=None,uvdist=False):
  # len(file_list)==len(SAP_list)
  # SAP_list should match each file name in file_list
  # open LOFAR H5 file, read data from a SAP,
  # randomly select number of baselines equal to batch_size
  # and sample patches and return input for training
  # num_channels=4 real,imag XX and YY
  # num_channels=8 real,imag XX, XY, YX and YY 
  # if transform (torchvision.transforms) is given (not None)
  # each baseline patches will be transformed, and appended to the original data
  # in other words, the number of patches output will be 2 times the original value
  # the original and transformed data will be grouped according to the baselines
  # if uvdist=True, return u,v distance in wavelengths (per each patch)
  # average value for the central frequency and start time of observation

  # light speed
  c=2.99792458e8

  assert(len(file_list)==len(SAP_list))
  assert(num_channels==4 or num_channels==8)
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

  # pad zeros if ntime or nfreq is smaller than patch_size
  x=torch.zeros(batch_size,num_channels,max(ntime,patch_size),max(nfreq,patch_size)).to(mydevice,non_blocking=True)
  # randomly select baseline subset
  baselinelist=np.random.randint(0,nbase,batch_size)

  if uvdist:
    # observation start time
    hms=f['measurement']['info']['start_time'][0].decode('ascii').split()[1].split(sep=':')
    # time in hours, in [0,24]
    start_time=float(hms[0])+float(hms[1])/60.0+float(hms[2])/3600
    # convert to radians
    theta=start_time/24.0*(2*math.pi)
    # frequencies in Hz
    frq=f['measurement']['saps'][SAP]['central_frequencies']
    Nf0=frq.shape[0]//2
    # central frequency
    freq0=frq[Nf0]
    # 1/lambda=freq0/c
    inv_lambda=freq0/c
    # rotation matrix =[cos(theta) sin(theta); -sin(theta) cos(theta)]
    rot00=math.cos(theta)*inv_lambda
    rot01=math.sin(theta)*inv_lambda

    baselines=f['measurement']['saps'][SAP]['baselines']
    xyz=f['measurement']['saps'][SAP]['antenna_locations']['XYZ']
    uv=torch.zeros(batch_size,2).to(mydevice,non_blocking=True)

  ck=0
  for mybase in baselinelist:
   if num_channels==8:
     # this is 8 channels in torch tensor
     for ci in range(4):
      # get visibility scales
      scalefac=torch.from_numpy(h[mybase,:,ci]).to(mydevice,non_blocking=True)
      # add missing (time) dimension
      scalefac=scalefac[None,:]
      x[ck,2*ci,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,0])
      x[ck,2*ci,:ntime,:nfreq]=x[ck,2*ci,:ntime,:nfreq]*scalefac
      x[ck,2*ci+1,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,1])
      x[ck,2*ci+1,:ntime,:nfreq]=x[ck,2*ci+1,:ntime,:nfreq]*scalefac
   else: # num_channels==4
      ci=0
      # get visibility scales
      scalefac=torch.from_numpy(h[mybase,:,ci]).to(mydevice,non_blocking=True)
      # add missing (time) dimension
      scalefac=scalefac[None,:]
      x[ck,0,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,0])
      x[ck,0,:ntime,:nfreq]=x[ck,0,:ntime,:nfreq]*scalefac
      x[ck,1,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,1])
      x[ck,1,:ntime,:nfreq]=x[ck,1,:ntime,:nfreq]*scalefac
      ci=3
      scalefac=torch.from_numpy(h[mybase,:,ci]).to(mydevice,non_blocking=True)
      scalefac=scalefac[None,:]
      x[ck,2,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,0])
      x[ck,2,:ntime,:nfreq]=x[ck,2,:ntime,:nfreq]*scalefac
      x[ck,3,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,1])
      x[ck,3,:ntime,:nfreq]=x[ck,3,:ntime,:nfreq]*scalefac

   if uvdist:
     # get u,v coordinates for this baseline
     # convert xx,yy to wavelengths and rotate by theta
     xx=xyz[baselines[mybase][0]][0]-xyz[baselines[mybase][1]][0]
     yy=xyz[baselines[mybase][0]][1]-xyz[baselines[mybase][1]][1]
     uu=xx*rot00+yy*rot01
     vv=-xx*rot01+yy*rot00
     uv[ck,0]=uu
     uv[ck,1]=vv

   ck=ck+1


  #torchvision.utils.save_image(x[0,0].data, 'sample.png')
  stride = patch_size//2 # patch stride (with 1/2 overlap)
  y = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
  # get new shape
  (nbase1,nchan1,patchx,patchy,nx,ny)=y.shape
  # create a new tensor
  y1=torch.zeros([nbase1*patchx*patchy,nchan1,nx,ny]).to(mydevice,non_blocking=True)

  if uvdist:
    # create a tensor for uv coordinates to match size of y1
    uv1=torch.zeros([nbase1*patchx*patchy,2]).to(mydevice,non_blocking=True)

  # copy data ordered according to the patches
  ck=0
  for ci in range(patchx):
   for cj in range(patchy):
     y1[ck*nbase1:(ck+1)*nbase1,:,:,:]=y[:,:,ci,cj,:,:]
     ck=ck+1

  if uvdist:
    for ci in range(nbase1):
     uv1[ci*patchx*patchy:(ci+1)*patchx*patchy,0]=uv[ci,0]
     uv1[ci*patchx*patchy:(ci+1)*patchx*patchy,1]=uv[ci,1]

  y = y1
  del x,y1
  # note: nbatch = batch_size x patchx x patchy
  #(nbatch,nchan,nxx,nyy)=y.shape

  # do some rough cleanup of data
  ##y[y!=y]=0 # set NaN,Inf to zero
  y.clamp_(-1e3,1e3) # clip high values

  # normalize data
  if normalize_data:
   ymean=y.mean()
   ystd=y.std()
   y.sub_(ymean).div_(ystd)

  # transform
  if transform:
    # create empty data (first dimension will be 2x  original)
    y1=torch.zeros(2*nbase1*patchx*patchy,nchan1,nx,ny).to(mydevice,non_blocking=True)
    # interleave original and transformed data according to the baseline
    for ci in range(nbase1):
      y1[2*ci*patchx*patchy:(2*ci+1)*patchx*patchy]=y[ci*patchx*patchy:(ci+1)*patchx*patchy]
      y1[(2*ci+1)*patchx*patchy:(2*ci+2)*patchx*patchy]=transform(y[ci*patchx*patchy:(ci+1)*patchx*patchy])
    y=y1

  # Note: if transform is given, size of y is doubled
  # size y: batchsize,channels,patch_size,patch_size
  # size uv1: batchsize,2
  if uvdist:
    return patchx,patchy,y,uv1
  else:
    return patchx,patchy,y

########################################################
def get_data_for_baseline(filename,SAP,baseline_id,patch_size=32,num_channels=8,give_baseline=False,uvdist=False):
  # open LOFAR H5 file, read data from a SAP,
  # return data for given baseline_id
  # num_channels=4 real,imag XX and YY
  # num_channels=8 real,imag XX, XY, YX and YY 
  # if give_basline=True, also return tuple [station1,station2] of the selected baseline
  # if uvdist=True, return u,v distance in wavelengths (per each patch)
  # average value for the central frequency and start time of observation

  # light speed
  c=2.99792458e8

  assert(num_channels==4 or num_channels==8)
  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']
  # scale factors for the dataset (float32)
  h=f['measurement']['saps'][SAP]['visibility_scale_factors']
  if give_baseline or uvdist:
    baselines=f['measurement']['saps'][SAP]['baselines']

  (nbase,ntime,nfreq,npol,ncomplex)=g.shape
  # h shape : nbase, nfreq, npol

  # pad zeros if ntime or nfreq is smaller than patch_size
  x=torch.zeros(1,num_channels,max(ntime,patch_size),max(nfreq,patch_size))
  
  mybase=baseline_id
  if uvdist:
    # observation start time
    hms=f['measurement']['info']['start_time'][0].decode('ascii').split()[1].split(sep=':')
    # time in hours, in [0,24]
    start_time=float(hms[0])+float(hms[1])/60.0+float(hms[2])/3600
    # convert to radians
    theta=start_time/24.0*(2*math.pi)
    # frequencies in Hz
    frq=f['measurement']['saps'][SAP]['central_frequencies']
    Nf0=frq.shape[0]//2
    # central frequency
    freq0=frq[Nf0]
    # 1/lambda=freq0/c
    inv_lambda=freq0/c
    # rotation matrix =[cos(theta) sin(theta); -sin(theta) cos(theta)]
    rot00=math.cos(theta)*inv_lambda
    rot01=math.sin(theta)*inv_lambda

    baselines=f['measurement']['saps'][SAP]['baselines']
    xyz=f['measurement']['saps'][SAP]['antenna_locations']['XYZ']
    uv=torch.zeros(1,2).to(mydevice,non_blocking=True)

  # this is 8 channels in torch tensor
  if num_channels==8:
   for ci in range(4):
    # get visibility scales
    scalefac=torch.from_numpy(h[mybase,:,ci])
    # add missing (time) dimension
    scalefac=scalefac[None,:]
    x[0,2*ci,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[0,2*ci,:ntime,:nfreq]=x[0,2*ci,:ntime,:nfreq]*scalefac
    x[0,2*ci+1,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[0,2*ci+1,:ntime,:nfreq]=x[0,2*ci+1,:ntime,:nfreq]*scalefac
  else: # num_channels==4
    ci=0
    # get visibility scales
    scalefac=torch.from_numpy(h[mybase,:,ci])
    # add missing (time) dimension
    scalefac=scalefac[None,:]
    x[0,0,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[0,0,:ntime,:nfreq]=x[0,0,:ntime,:nfreq]*scalefac
    x[0,1,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[0,1,:ntime,:nfreq]=x[0,1,:ntime,:nfreq]*scalefac
    ci=3
    scalefac=torch.from_numpy(h[mybase,:,ci])
    scalefac=scalefac[None,:]
    x[0,2,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[0,2,:ntime,:nfreq]=x[0,2,:ntime,:nfreq]*scalefac
    x[0,3,:ntime,:nfreq]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[0,3,:ntime,:nfreq]=x[0,3,:ntime,:nfreq]*scalefac

  if uvdist:
     # get u,v coordinates for this baseline
     # convert xx,yy to wavelengths and rotate by theta
     xx=xyz[baselines[mybase][0]][0]-xyz[baselines[mybase][1]][0]
     yy=xyz[baselines[mybase][0]][1]-xyz[baselines[mybase][1]][1]
     uu=xx*rot00+yy*rot01
     vv=-xx*rot01+yy*rot00
     uv[0,0]=uu
     uv[0,1]=vv

  stride = patch_size//2 # patch stride (with 1/2 overlap)
  y = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
  # get new shape
  (nbase1,nchan1,patchx,patchy,nx,ny)=y.shape
  # create a new tensor
  y1=torch.zeros([nbase1*patchx*patchy,nchan1,nx,ny]).to(mydevice,non_blocking=True)

  if uvdist:
    # create a tensor for uv coordinates to match size of y1
    uv1=torch.zeros([nbase1*patchx*patchy,2]).to(mydevice,non_blocking=True)

  # copy data ordered according to the patches
  ck=0
  for ci in range(patchx):
   for cj in range(patchy):
     y1[ck*nbase1:(ck+1)*nbase1,:,:,:]=y[:,:,ci,cj,:,:]
     ck=ck+1

  if uvdist:
    for ci in range(nbase1):
      uv1[ci*patchx*patchy:(ci+1)*patchx*patchy,0]=uv[ci,0]
      uv1[ci*patchx*patchy:(ci+1)*patchx*patchy,1]=uv[ci,1]

  y = y1
  del x,y1
  # note: nbatch = batch_size x patchx x patchy
  #(nbatch,nchan,nxx,nyy)=y.shape

  # do some rough cleanup of data
  ##y[y!=y]=0 # set NaN,Inf to zero
  y.clamp_(-1e6,1e6) # clip high values

  # normalize data
  ymean=y.mean()
  ystd=y.std()
  y.sub_(ymean).div_(ystd)

  if not give_baseline:
    if uvdist:
      return patchx,patchy,y,uv1
    else:
      return patchx,patchy,y
  else:
    if uvdist:
      return baselines[mybase],patchx,patchy,y,uv1
    else:
      return baselines[mybase],patchx,patchy,y

########################################################
def get_data_for_baseline_flat(filename,SAP,baseline_id,patch_size=32,num_channels=8):
  # open LOFAR H5 file, read data from a SAP,
  # return data for given baseline_id
  # Note : 'without unfolding' 
  # num_channels=4 real,imag XX and YY
  # num_channels=8 real,imag XX, XY, YX and YY 
  assert(num_channels==4 or num_channels==8)

  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']
  # scale factors for the dataset (float32)
  h=f['measurement']['saps'][SAP]['visibility_scale_factors']

  (nbase,ntime,nfreq,npol,ncomplex)=g.shape
  # h shape : nbase, nfreq, npol

  x=torch.zeros(1,num_channels,ntime,nfreq)
  
  mybase=baseline_id
  if num_channels==8:
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
  else: # num_channels==4
    ci=0
    # get visibility scales
    scalefac=torch.from_numpy(h[mybase,:,ci])
    # add missing (time) dimension
    scalefac=scalefac[None,:]
    x[0,0]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[0,0]=x[0,0]*scalefac
    x[0,1]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[0,1]=x[0,1]*scalefac
    ci=3
    scalefac=torch.from_numpy(h[mybase,:,ci])
    scalefac=scalefac[None,:]
    x[0,2]=torch.from_numpy(g[mybase,:,:,ci,0])
    x[0,2]=x[0,2]*scalefac
    x[0,3]=torch.from_numpy(g[mybase,:,:,ci,1])
    x[0,3]=x[0,3]*scalefac


  # do some rough cleanup of data
  ##y[y!=y]=0 # set NaN,Inf to zero
  x.clamp_(-1e6,1e6) # clip high values

  return x


########################################################
def get_metadata(filename,SAP,give_baseline=False):
  # open LOFAR H5 file, read metadata from a SAP,
  # return number of baselines, time, frequencies, polarizations, real/imag
  # if give_baseline=True, also return ndarray of baselines

  f=h5py.File(filename,'r')
  # select a dataset SAP (int8)
  g=f['measurement']['saps'][SAP]['visibilities']

  if give_baseline:
    baselines=f['measurement']['saps'][SAP]['baselines']
    (nbase,_)=baselines.shape
    bline=np.ndarray(baselines.shape,dtype=object)
    for ci in range(nbase):
      bline[ci]=baselines[ci]
    return bline,g.shape
  return g.shape
 

########################################################
def get_fileSAP(pathname,pattern='L*.MS_extract.h5'):
  # search in pathname for files matching 'pattern'
  # test valid SAPs in each file and
  # return file_list,sap_list for valid files and their SAPs
  file_list=[]
  sap_list=[]
  if rec_file_search:
    rawlist = glob.glob(pathname+'**'+os.sep+pattern,recursive=True)
  else:
    rawlist=glob.glob(pathname+os.sep+pattern)
  # open each file and check valid saps
  for filename in rawlist:
    f=h5py.File(filename,'r')
    g=f['measurement']['saps']
    SAPs=[SAP for SAP in g]
    # flag to remember if this file is useful
    fileused=False
    if len(SAPs)>0:
     for SAP in SAPs:
      try:
       vis=f['measurement']['saps'][SAP]['visibilities']
       (nbase,ntime,nfreq,npol,reim)=vis.shape
       # select valid datasets (larger than 90 say)
       if nbase>1 and nfreq>=90 and ntime>=90 and npol==4 and reim==2:
         file_list.append(filename)
         sap_list.append(SAP)
         fileused=True
      except:
       print('Failed opening'+filename)
    
    if not fileused:
      print('File '+filename+' not used') 

  return file_list,sap_list
########################################################
