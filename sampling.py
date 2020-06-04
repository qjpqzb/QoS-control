# -*- coding: utf-8 -*-
from __future__ import division
import rados
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
np.set_printoptions(threshold=np.inf)
def find_large_elements(source,threshold):
    s_mean = np.mean(source)
    s_std = np.std(source,ddof=1)
    high = s_mean + 3.5 * s_std * threshold
    low = s_mean - 3.5 * s_std * threshold
    return high,low

def find_augment_points(base,chosn_index,deci_ratio,threshold):
    if threshold == 0.0:
        return [range(len(base))]
    elif threshold == 1.0:
        return []
    chosn_points=[]

    delta_temp=[]
    temp_index=[]
    temp_interval=[]
    chosn_index_finer=[]
    base_gradient=np.gradient(base)
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
                chosn_points.append(base_gradient[j])
   
    high_b, low_b = find_large_elements(chosn_points, threshold)
    #print high_b, low_b
    #uplimit = quantile(chosn_points,1.5)   
    #uplimit=outlier(chosn_delta)
    #print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
            if base_gradient[j]>=high_b or base_gradient[j]<low_b:
                #print j
                temp_1.append(j)
        if len(temp_1)>1:
            temp_index.append(temp_1)
        temp_1=[]
                                          
    for i in temp_index:
        for j in range(1,len(i)):
            if i[j]-i[j-1]>1:
                temp_interval.append(i[j]-i[j-1]) 
    
    #print "temp_interval",temp_interval
    max_intv = k_means(temp_interval,'false', 'false')

    #max_intv=quantile(temp_interval,1.5)
    print "max_intv=",max_intv
    #print temp_index                  
    temp_2=[]

    for i in temp_index:
        temp_2.append(i[0])
        for j in range(1,len(i)):
            if i[j]-i[j-1] <= max_intv:
                temp_2.append(i[j])
            else:
                #print temp_2
                if len(temp_2)>1:
                    chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
                temp_2=[]
                temp_2.append(i[j])
        if len(temp_2)>1:
            chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
        temp_2=[]
    #cnt=0
    #for i in range(len(chosn_index_finer)):
    #    for j in range(len(chosn_index_finer[i])):
    #        cnt+=1   
    #print "number of chosn index=",cnt
    #print np.shape(chosn_index_finer)
    return chosn_index_finer

def partial_refinement(chosn_index, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer=np.zeros((len(base)-1)*deci_ratio+1)
    finer_r=np.zeros((len(base)-1)*deci_ratio+1)
    finer_z=np.zeros((len(base)-1)*deci_ratio+1)
    #finer_p=[]
    #finer_r_p=[]
    #finer_z_p=[]
    #print chosn_index
    start=0
    inc = 0
    i=0
    num_refine=0
    while(i<len(chosn_index)):
        for m in range(start,chosn_index[i]):
            index1 = m // deci_ratio
            index2 = m % deci_ratio
            if index1!=len(base)-1:
                if index2!=0:
                    finer[m]=(base[index1]+base[index1+1])*index2/deci_ratio
                    finer_r[m]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                    finer_z[m]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                else:
                    finer[m]=base[index1]
                    finer_r[m]=base_r[index1]
                    finer_z[m]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[m]=2*base[index1]*index2/deci_ratio
                    finer_r[m]=2*base_r[index1]*index2/deci_ratio
                    finer_z[m]=2*base_z[index1]*index2/deci_ratio
                else:
                    finer[m]=base[index1]
                    finer_r[m]=base_r[index1]
                    finer_z[m]=base_z[index1]
        for n in range(chosn_index[i],chosn_index[i+1]+1):
            index1 = n // deci_ratio
            index2 = n % deci_ratio
            if index1!=len(base)-1:
                if index2!=0:
                    #print i,k
                    finer[n]=(base[index1]+base[index1+1])*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio+chosn_z[inc]
                    num_refine+=1

                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[n]=2*base[index1]*index2/deci_ratio+chson_data[i][inc]
                    finer_r[n]=2*base_r[index1]*index2/deci_ratio+chson_r[i][inc]
                    finer_z[n]=2*base_z[index1]*index2/deci_ratio+chson_z[i][inc]   
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            inc+=1
        start = chosn_index[i+1]+1
        if i == len(chosn_index)-2:
            for j in range(start,(len(base)-1)*deci_ratio+1):
                index1 = j // deci_ratio
                index2 = j % deci_ratio
                if index1!=len(base)-1:
                    if index2!=0:
                        finer[j]=(base[index1]+base[index1+1])*index2/deci_ratio
                        finer_r[j]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                        finer_z[j]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                    else:
                        finer[j]=base[index1]
                        finer_r[j]=base_r[index1]
                        finer_z[j]=base_z[index1]
                        #finer_p.append(base[index1])
                        #finer_r_p.append(base_r[index1])
                        #finer_z_p.append(base_z[index1])
                else:
                    if index2!=0:
                        finer[j]=2*base[index1]*index2/deci_ratio
                        finer_r[j]=2*base_r[index1]*index2/deci_ratio
                        finer_z[j]=2*base_z[index1]*index2/deci_ratio
                    else:
                        finer[j]=base[index1]
                        finer_r[j]=base_r[index1]
                        finer_z[j]=base_z[index1]

                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
        i+=2
    print "percentage of selected points= ", num_refine/(len(delta_L1)-len(base))
    #for i in chosn_index:
    #    if i[-1] != len(base)-1:
    #        finer_chosn_index.append(range(i[0]*deci_ratio,(i[-1]+1)*deci_ratio))
    #    else:
    #        finer_chosn_index.append(range(i[0]*deci_ratio,len(delta_L1)))
    #PSNR=psnr(delta_L1,chosn_index_n,np.max(finer),len(delta_L1))
    #print finer
    #print 3
    return finer, finer_r, finer_z

def k_means(source,savefig,k):
    if k == "true":
        find_k(source,savefig)
    y=np.array(source).reshape(-1,1)
    km=KMeans(n_clusters=5)
    km.fit(y)
    km_label=km.labels_
    #print "source=",source
    #print "km.label=",km_label
    #print km.cluster_centers_
    if len(km_label)!=len(source):
        print "length issue"
    sorted_cluster_index=np.argsort(km.cluster_centers_.reshape(-1,))
    #print "sorted_cluster_index=",sorted_cluster_index
    group=sorted_cluster_index[:1]
    #print "group=",group
    group_index=[]

    for i in range(len(km_label)):
        if km_label[i] in group: group_index.append(i)
    limit = 0
    for i in range(len(source)):

        if km_label[i] in group:
            #print km_label[i]
            if source[i]>limit:
                #print source[i],limit
                limit = source[i]
    return limit

def noise_threshold(noise):
    if noise > 0.75:
        threshold = 0.0
    elif noise < 0.1:
        threshold = 1.0 
    else:
        k=(1.0-0.0)/(0.1-0.75)
        b =1.0-k*0.1
        threshold = noise*k+b

    return threshold

def FFT (Fs,data):
    L = len (data)
    N =int(np.power(2,np.ceil(np.log2(L))))
    FFT_y = np.abs(fft(data,N))/L*2 
    Fre = np.arange(int(N/2))*Fs/N
    FFT_y = FFT_y[range(int(N/2))]
    return Fre, FFT_y

def plot(data,r,z):
    points = np.transpose(np.array([z,r]))
    print points
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices

	#plt.figure(206, figsize=(10,7))
    fig,ax=plt.subplots(figsize=(8,8))
    plt.rc('xtick', labelsize=26)          # fontsize of the tick labels
    plt.rc('ytick', labelsize=26)  

    axis_font = {'fontname':'Arial', 'size':'38'}

    #plt.xlabel('R', **axis_font)
   #plt.ylabel('Z',**axis_font )
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=250));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
    	    spine.set_visible(False)
    plt.savefig('w_finer.png', format='png')

def whole_refine(base,base_r,base_z,delta_L1,delta_L1_r,delta_L1_z,deci_ratio):
    finer=np.zeros(len(delta_L1))
    finer_r=np.zeros(len(delta_L1))
    finer_z=np.zeros(len(delta_L1))

    for i in range(len(delta_L1)):
        index1 = i//deci_ratio
        index2 = i%deci_ratio
        if index1!=len(base)-1:
            if index2!=0:
                finer[i]=(base[index1]+base[index1+1])*index2/deci_ratio
                finer_r[i]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                finer_z[i]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
            else:
                finer[i]=base[index1]
                finer_r[i]=base_r[index1]
                finer_z[i]=base_z[index1]
        else:
            if index2!=0:
                finer[i]=2*base[index1]*index2/deci_ratio
                finer_r[i]=2*base_r[index1]*index2/deci_ratio
                finer_z[i]=2*base_z[index1]*index2/deci_ratio
            else:
                finer[i]=base[index1]
                finer_r[i]=base_r[index1]
                finer_z[i]=base_z[index1]
    return finer,finer_r,finer_z

def sample_read(Nsamples):
    print "123"
    start = time.time()
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
	#finally:
	   #print "Connected to the cluster."
    print cluster.conf_get("bluestore default buffered read")
    if not cluster.pool_exists('tier2_pool'):
    	raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')
    oname='all_diff'
    print "Object name:",oname
    print ioctx_2.stat(oname)[0]/1024/1024
    sum_time = 0.0
    #a =time.time()
    #op=ioctx_2.create_read_op()
    for i in range(Nsamples):
        a =time.time()
        #if i%2 == 1: 
        	#test_sample_str = ioctx_2.read("all_same",ioctx_2.stat("all_same")[0],0)
        #else:
        #op=ioctx_2.create_read_op()
        #test_sample_str=ioctx_2.operate_read_op(op, oname,8)
        test_sample_str = ioctx_2.read(oname,ioctx_2.stat(oname)[0],0)
		#tag = str(int(ioctx_2.stat("test_sample")[0]/8))
		#delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)
        
        e=time.time()
        #sum_time+=e-a
        print i,e-a
        y.append(e-a)
    #ioctx_2.release_read_op(op)
    #e=time.time()
    #print e-a
    ioctx_2.close()
    cluster.shutdown()
    #print "Average time",sum_time/Nsamples
    prefix_time=""
    for i in range(len(y)-1):
    	prefix_time += str(y[i])+","
    prefix_time += str(y[-1])
    file=open("result_fully.txt","w")
    file.write(prefix_time)
    file.close()
    end = time.time()
    print "Sampling finished, takes",end-start
    
def interference(cmd):
    nowtime = os.popen(cmd)
    print nowtime.read()

#file=open("result_fully.txt","a")
prefix_time=""
start = time.time()
s_refine_t_w=0
y=[]
sample_rate = 2
Nsamples = 256
print "start sampling\n"
sample_read(Nsamples)
