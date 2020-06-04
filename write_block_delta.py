from __future__ import division
import rados
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import triangulation
from scipy.spatial import Delaunay
from scipy import stats
import numpy as np
import math
import time
import struct
import sys
import os
import subprocess
from sklearn.cluster import KMeans
#from scipy.fftpack import fft
import scipy.fftpack as fft
import Queue
from threading import Thread
import zfpy
import argparse
import cv2
np.set_printoptions(threshold=np.inf)

def find_large_elements(source,block):
    
    max_gradient=0.0
    for i in range(len(source)):
        if math.fabs(source[i])> max_gradient:
            max_gradient = math.fabs(source[i])

    return max_gradient * block[0], max_gradient * block[1]  

def check_duplicated_elements(source):
    seen = set()
    duplicated = set()
    for x in source:  
        if x not in seen:  
            seen.add(x)
        else:
            duplicated.add(x)
    print duplicated

reduced_len = 4876
filename = "../reduced_data.bin"
f = open(filename, "rb")
dpot_L1_str=f.read(reduced_len*8)
#dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
r_L1_str=f.read(reduced_len*8)
#r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
z_L1_str=f.read(reduced_len*8)
#z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
f.close()
dpot_L1=struct.unpack(str(reduced_len)+'d',dpot_L1_str)
r_L1=struct.unpack(str(reduced_len)+'d',r_L1_str)
z_L1=struct.unpack(str(reduced_len)+'d',z_L1_str)

try:
    cluster = rados.Rados(conffile='')
except TypeError as e:
    print 'Argument validation error: ', e
    raise e

try:
    cluster.connect()
except Exception as e:
    print "connection error: ", e
    raise e

if not cluster.pool_exists('tier2_pool'):
    raise RuntimeError('No data pool exists')
ioctx_2 = cluster.open_ioctx('tier2_pool')
delta_L0_L1_str = ioctx_2.read("delta_L0_L1_o",ioctx_2.stat("delta_L0_L1_o")[0],0)
delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1_o",ioctx_2.stat("delta_r_L0_L1_o")[0],0)
delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1_o",ioctx_2.stat("delta_z_L0_L1_o")[0],0)
delta_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_L0_L1_o")[0]/8))+'d',delta_L0_L1_str)
delta_r_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_r_L0_L1_o")[0]/8))+'d',delta_r_L0_L1_str)
delta_z_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_z_L0_L1_o")[0]/8))+'d',delta_z_L0_L1_str)
print ioctx_2.stat("delta_L0_L1_o")[0]/8
base_gradient=np.fabs(np.gradient(dpot_L1))
print len(base_gradient)
#sorted_gradient = sorted(base_gradient)
sorted_index = np.argsort(base_gradient)
number_blocks = 50
print "number_blocks=",number_blocks
interval = len(base_gradient)//number_blocks
remain = len(base_gradient)% number_blocks
blocklist=range(0,len(base_gradient),interval)
blocklist[-1] = blocklist[-1] + remain -1
print blocklist
print len(blocklist)
deci_ratio = 1024
print len(dpot_L1)
delta_index_total = []
block_interval = 1/number_blocks
for i in range(len(blocklist)-1):
    #index=[]
    delta_index = []
    delta=[]
    delta_r=[]
    delta_z=[]
    #if you want to change number of blocks, remember to change block_interval below to let block name between 0 and 1
    tag = block_interval*(len(blocklist)-2-i)
    print tag
    if i !=len(blocklist)-2:
        index = sorted_index[blocklist[i]:blocklist[i+1]]
    else:
        index = sorted_index[blocklist[i]:]
    print len(index)
    index = np.sort(index)
    first = index[0]
    last_tag = index[0]
    delta_index = []
    for j in index:
        if j-last_tag > 1: 
            delta_index +=range(first*deci_ratio, (last_tag+1)*deci_ratio)
            first = j
        if j == len(dpot_L1)-1:
            delta_index +=range(first*deci_ratio, len(delta_L0_L1))
        last_tag = j
    if index[-1] !=len(dpot_L1)-1:
        delta_index +=range(first*deci_ratio, (last_tag+1)*deci_ratio)
    for j in delta_index:
        delta.append(delta_L0_L1[j])
        delta_r.append(delta_r_L0_L1[j])
        delta_z.append(delta_z_L0_L1[j])
    
    chosn_delta_name= "delta_L0_L1_"+str(tag)
    chosn_delta_r_name = "delta_r_L0_L1_"+str(tag)  
    chosn_delta_z_name = "delta_z_L0_L1_"+str(tag)    
    chosn_index_name = "delta_index_" + str(tag)
    #print chosn_delta_name, chosn_delta_r_name, chosn_delta_z_name, chosn_index_name
    #print len(delta), len(delta_r), len(delta_z), len(delta_index)
    ioctx_2.write_full(chosn_index_name, struct.pack(str(len(delta_index))+'i',*delta_index))    
    ioctx_2.write_full(chosn_delta_name, struct.pack(str(len(delta))+'d',*delta))
    ioctx_2.write_full(chosn_delta_r_name, struct.pack(str(len(delta_r))+'d',*delta_r))
    ioctx_2.write_full(chosn_delta_z_name, struct.pack(str(len(delta_z))+'d',*delta_z))
    delta_index_total+=delta_index 
ioctx_2.close()
cluster.shutdown()

#for i in range(len(delta_L0_L1)):
#    if i not in delta_index_total:
#        print i            

