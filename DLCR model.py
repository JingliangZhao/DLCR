# Technical details of the DLCR network
# This code is run with pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

import ast
import numpy as np
from operator import itemgetter, attrgetter
from numpy import pi, cos, sin, arccos, arange
from collections import namedtuple
from skimage import transform
from scipy.ndimage import zoom



# Models for the DRL agent
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        
        input_channel = 1
        init_channels = 32
        output_num = 100
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, init_channels, kernel_size=3),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Conv2d(init_channels, init_channels, kernel_size=3),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Dropout(p = 0.35),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Conv2d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Dropout(p = 0.35),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(init_channels, init_channels*2, kernel_size=3),
            nn.BatchNorm2d(init_channels*2),
            nn.ReLU(),
            
            nn.Conv2d(init_channels*2, output_num, kernel_size=1),
            nn.BatchNorm2d(output_num),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size = output_num, hidden_size = 500, num_layers=1, batch_first=True)
        
        self.dp = nn.Dropout(p=0.35)
        
        self.fc = nn.Sequential(
            nn.Linear(500, 100),
            nn.Sigmoid()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(100, 2),
            nn.Sigmoid()
        )
    
    
    def forward(self, x):
        # the input size should be：[batch, 5, 32, 32]
        ts=[]   # the vector passs to LSTM
        for i in range(0, 5):
            y = x[:,i,:,:].unsqueeze(dim = 1)
            
            t = self.features( y )
            t2 = t[:,:, 0, 0].unsqueeze(dim = 1)
            ts.append( t2 )
        
        ts2 = torch.cat(tuple(ts), 1)
        
        lstm_out, _ = self.lstm(ts2)
        pred = self.fc(lstm_out)
        
        pred_2 = self.fc2(pred)         # size of pred_2 : [batch, 5, 2]
        
        return pred_2
 
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        input_channel = 1
        init_channels = 32
        output_num = 100
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, init_channels, kernel_size=3),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Conv2d(init_channels, init_channels, kernel_size=3),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Dropout(p = 0.35),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Conv2d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Dropout(p = 0.35),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(init_channels, init_channels*2, kernel_size=3),
            nn.BatchNorm2d(init_channels*2),
            nn.ReLU(),
            
            nn.Conv2d(init_channels*2, output_num, kernel_size=1),
            nn.BatchNorm2d(output_num),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size = output_num + 2, hidden_size = 500, num_layers=1, batch_first=True)
        
        self.dp = nn.Dropout(p=0.35)
        
        self.fc = nn.Sequential(
            nn.Linear(500, 100),
            nn.Sigmoid()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    
    
    def forward(self, x, action):
        ts=[]
        for i in range(0, 5):
            y = x[:,i,:,:].unsqueeze(dim = 1)
            
            t = self.features( y )
            t2 = t[:,:, 0, 0].unsqueeze(dim = 1)
            ts.append( t2 )
        
        ts = torch.cat(tuple(ts), 1)
        
        ts2 = torch.cat([ts, action], 2)
        
        lstm_out, _ = self.lstm(ts2)
        pred = self.fc(lstm_out)
        
        value = self.fc2(pred)
        value = torch.mean(value, dim=1, keepdim=False)
        
        return value

# Model for VS-net
class Reg_wd(nn.Module):
    
    def __init__(self):
        super(Reg_wd, self).__init__()
        
        input_channel = 1
        init_channels = 32
        output_num = 1
        
        self.features = nn.Sequential(
            nn.Conv3d(input_channel, init_channels, kernel_size=3),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),
            
            nn.Conv3d(init_channels, init_channels, kernel_size=3),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),
            
            nn.Dropout(p = 0.35),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),
            
            nn.Conv3d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),
            
            nn.Dropout(p = 0.35),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(init_channels, init_channels*2, kernel_size=3),
            nn.BatchNorm3d(init_channels*2),
            nn.ReLU(),
            
            nn.Conv3d(init_channels*2, init_channels*2, kernel_size=1),
            nn.BatchNorm3d(init_channels*2),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Conv3d(init_channels*2, output_num, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # size of input should be : [batch, 1, 32, 32, 32]
        x = self.features(x)
        
        output = self.advantage(x)
        
        output = output.squeeze(-1).squeeze(-1).squeeze(-1)
        return output



# Pre-calculated masks to accelerate patch extraction 
dc_reg = np.load('dc.npy')

# Function to extract a 2.5D state
def get_state(device, img, pasi, dc_reg):
    
    # device    :   'cpu' or torch.device('cuda')
    # img       :   the CTA image
    # pasi      :   position, tangential vector, and scale of the center point
    # dc_reg    :   a pre-calculated mask to accelerate the process
    
    szx,szy,szz=img.shape
    pasi = np.array(pasi)
    
    pp0 = pasi[0:3]     # pt
    vi  = pasi[3:6]     # vt
    v1  = pasi[6:9]
    v2  = pasi[9:12]    # (v1, v2, vi) forms a local coordinate system
    wd  = pasi[12]      # wt, the vessel scale. The size of the receptive field is (wd*2 + 1)^2
    
    step = wd/12        # the distance between adjacent slices, which is also the tracking step length
    
    d  = 16     # sample (d*2 + 1) * (d*2 + 1) points as a slice. Regardless of wd.
    
    sp = wd/d   # the spacing of the sampling points
    
    v1 = sp*v1;   v2 = sp*v2
    
    xx=[];   yy=[];   zz=[]
    d1=dc_reg[0];   d2=dc_reg[1]   # pre-calculated mask for sampling
    
    # We extract 5 slices with j = -2, -1, 0, 1, 2
    for k in range(-2, 3):
        ppk = pp0 + step*k*vi   # the center of each slice
        
        xxk = np.round( ppk[0] + d1*v1[0] + d2*v2[0] )
        yyk = np.round( ppk[1] + d1*v1[1] + d2*v2[1] )
        zzk = np.round( ppk[2] + d1*v1[2] + d2*v2[2] )   # the coordinates of the sampling points
        
        xx = np.hstack( [xx, xxk] )
        yy = np.hstack( [yy, yyk] )
        zz = np.hstack( [zz, zzk] )
    
    xx[xx<0]=0;   xx[xx>szx-1]=szx-1
    yy[yy<0]=0;   yy[yy>szy-1]=szy-1
    zz[zz<0]=0;   zz[zz>szz-1]=szz-1   # make sure that the points are within the image
    
    ps = img[np.int16(xx), np.int16(yy), np.int16(zz)]   # extract the intensity of each point
    
    # Restore to image
    sz=2*d+1;   im_sz = sz*sz
    ims = np.zeros([5, 32, 32])
    
    for k in range(0, 5):
        t = ps[k*im_sz : (k+1)*im_sz]
        
        t2 = t.reshape([sz, sz])
        
        ims[k,:,:] = zoom(t2, [32/sz, 32/sz], order=0)
    
    ims = mat2gray_bk(ims)*2-1          # intensity normalization
    ims = np.expand_dims(ims, axis=0)
    
    return torch.from_numpy(ims).to(device, dtype=torch.float)

# Function for gradient-based tangential vector modification
def update_vt(pasi, grd):
    
    # grd: gradient image of the CTA scan
    
    pasi = np.array(pasi)
    
    p0 = pasi[0:3];   vv = pasi[3:6];   v1 = pasi[6:9];   v2 = pasi[9:12];   wd = pasi[12]
    vv = vv.reshape([1,3]);   v1 = v1.reshape([1,3]);   v2 = v2.reshape([1,3])
    
    num = 100   # generat 100 random vectories
    alf = np.random.rand(num) * (math.pi/4);   alf = alf.reshape([num,1])
    blt = np.random.rand(num) * (2*math.pi);   blt = blt.reshape([num,1])
    
    vs = np.cos(alf) * vv + np.sin(alf)*( np.cos(blt)*v1 + np.sin(blt)*v2 )
    
    lv = vs.shape[0]
    
    step = np.round(2 * wd/6 ).astype(np.int16)
    t = np.array(range(1, step)) * 0.5;   lt = len(t);   t = t.reshape(lt,1)
    
    ps_all = np.zeros([1,3]);   ps0 = np.tile(p0, (lt, 1))
    for k in range(0, lv):
        vk = vs[k,:];   vk = vk.reshape(1,3)
        ps2 = np.round( ps0 + np.dot(t, vk) ).astype(np.int16)   # [lt, 3]
        ps2 = ps_in_box(ps2, grd)
        
        ps_all = np.vstack([ps_all, ps2])
    
    ps_all = ps_all[1:,:].astype(np.int16)
    grds = grd[ps_all[:,0], ps_all[:,1], ps_all[:,2]]
    
    grds = grds.reshape(lv, lt)   # line=vector，colume=sample points on a vector
    
    gv = np.mean(grds, 1)
    mv = np.max(gv)
    gv = mv - gv +1;   gv = gv.reshape(1, lv)
    vv2 = np.dot(gv, vs);   vv2 = vv2/np.linalg.norm(vv2)
    vv2 = vv2.flatten()

    [v1, v2] = cross123(vv2)
    pasi[3:12] = np.hstack([vv2, v1, v2])
    
    return pasi





# Image intensity normalization
def mat2gray_bk(im):
    im = im.astype(float)
    m1=np.min(im);   m2=np.max(im)
    
    if m2-m1 < 1e-6:
        return im
    else:
        det = 1/(m2-m1)
        im2 = (im - m1)*det
        
        return im2

# To insure that ps is Inside img
def ps_in_box(ps, img):
    sx, sy, sz = img.shape
    x=ps[:,0];   y=ps[:,1];   z=ps[:,2];   lx=len(x)
    
    x[x<0]=0;   y[y<0]=0;   z[z<0]=0
    x[x>=sx]=sx-1;   y[y>=sy]=sy-1;   z[z>=sz]=sz-1
    
    ps2 = np.hstack([ x.reshape([lx,1]), y.reshape([lx,1]), z.reshape([lx,1]) ])
    
    return ps2



#


















#