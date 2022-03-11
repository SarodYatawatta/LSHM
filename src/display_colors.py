import torch
import torchvision
import numpy as np
import h5py

from lofar_models import *
from lofar_tools import *

# Show how the colormap of the image depends 
# on various parameters in the data

num_channels=4 
num_freq=256
num_time=128
# time coords (s) (within 0..24 hrs)
start_time=1000.0
end_time=1500.0
# convert to radians
x_time=np.linspace(start_time,end_time,num_time)*2*np.pi/(24*60*60)
# freq coords (Hz)
start_freq=110e6
end_freq=180e6
# 1/wavelenght=divide by 1/c (c=speed of light)
x_freq=np.linspace(start_freq,end_freq,num_freq)/3e8
x_t,x_f=np.meshgrid(x_time,x_freq)

# baseline uv coordinate (m)
uv=np.random.rand((2))*1e3
# rotate uv coordinate with time, scale with freq (wavelengths)
uprime=np.cos(uv[0]*x_t)+np.sin(uv[1]*x_t)
vprime=np.sin(-uv[0]*x_t)+np.cos(uv[1]*x_t)
uprime=uprime*x_f
vprime=vprime*x_f

# Gains for XX,YY = [XX_real,XX_imag,YY_real,YY_imag]
Gain=[0.4,0.0,0.4,0]
# source coordinates (l,m)
lm=[0.5,-0.5]

y=np.zeros((1,num_channels,num_freq,num_time),dtype=float)
y[0,0]=Gain[0]*np.cos(uprime*lm[0]+vprime*lm[1])
y[0,1]=Gain[1]*np.sin(uprime*lm[0]+vprime*lm[1])
y[0,2]=Gain[2]*np.cos(uprime*lm[0]+vprime*lm[1])
y[0,3]=Gain[3]*np.sin(uprime*lm[0]+vprime*lm[1])

# transpose
y=torch.from_numpy(y)
y=torch.transpose(y,2,3)

x=channel_to_rgb(y[0])
torchvision.utils.save_image(x.data, 'xx_.png' )
