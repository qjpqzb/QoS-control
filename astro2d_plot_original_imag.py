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
from scipy.spatial import Delaunay
np.set_printoptions(threshold=np.inf)

def plot(data,r,z,filename):
    points = np.transpose(np.array([z,r]))
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices
    fig,ax=plt.subplots(figsize=(8,8))
    plt.rc('xtick', labelsize=26)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)

    axis_font = {'fontname':'Arial', 'size':'38'}

    #plt.xlabel('R', **axis_font)
    #plt.ylabel('Z',**axis_font )
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=25));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    plt.savefig(filename, format='png')

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

data_str = ioctx_2.read("data_o",ioctx_2.stat("data_o")[0],0)
r_str = ioctx_2.read("r_o",ioctx_2.stat("r_o")[0],0)
z_str = ioctx_2.read("z_o",ioctx_2.stat("z_o")[0],0)

data = struct.unpack(str(int(ioctx_2.stat("data_o")[0]/8))+'d',data_str)
r = struct.unpack(str(int(ioctx_2.stat("r_o")[0]/8))+'d',r_str)
z = struct.unpack(str(int(ioctx_2.stat("z_o")[0]/8))+'d',z_str)
ioctx_2.close()
cluster.shutdown()
start = time.time()
plot(data,r,z,"astro2d/original_no_upsampling.png")
end = time.time()
print "Plot time = ", end -start


