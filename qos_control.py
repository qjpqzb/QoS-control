# -*- coding: utf-8 -*-
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
def find_large_elements(source,threshold):
    max_gradient=0.0
    for i in range(len(source)):
        if math.fabs(source[i])> max_gradient:
            max_gradient = math.fabs(source[i])
     
    #s_mean = np.mean(source)
    #s_std = np.std(source,ddof=1)
    #print "source=",source
    #fig,ax = plt.subplots(figsize=(11,6))
    #if np.fabs(np.max(source)-s_mean) < np.fabs(np.min(source)-s_mean):
    #    interval = np.fabs(np.min(source)-s_mean)
    #else:
    #    interval = np.fabs(np.max(source)-s_mean)
    #plt.plot(source)
    #plt.hlines(s_mean + interval, 0, 2500000, 'r')
    #plt.hlines(s_mean - interval, 0, 2500000, 'b')
    #plt.savefig("source.pdf",format='pdf')
    
    #print threshold
    #high = s_mean + interval * threshold
    #low = s_mean - interval * threshold
    return max_gradient * threshold 

def find_augment_points_gradient(base,chosn_index,threshold):
    if threshold == 0.0:
        return [range(len(base))]
    elif threshold == 1.0:
        return []
    chosn_points=[]

    delta_temp=[]
    temp_index=[]
    temp_interval=[]
    chosn_index_finer=[]
    #print "-1\n"
    #print sys.getsizeof(base)/1024/1024
    base_gradient=np.gradient(base)
    #print "0\n"
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
                chosn_points.append(base_gradient[j])
    #print "1\n"   
    thre = find_large_elements(chosn_points, threshold)
    #print "2\n"  
    #print high_b, low_b
    #uplimit = quantile(chosn_points,1.5)   
    #uplimit=outlier(chosn_delta)
    #print "uplimit=",uplimit
    temp_1=[]
    for i in chosn_index:
        for j in range(i[0],i[-1]+1):
            if math.fabs(base_gradient[j])> thre:
                #print "VIP=",j
                temp_1.append(j)
        #if len(temp_1)>1:
        temp_index.append(temp_1)
        temp_1=[]
    #print "temp_index=",temp_index                                          
    for i in temp_index:
        if len(i)>1:
            for j in range(1,len(i)):
                if i[j]-i[j-1]>1:
                    temp_interval.append(i[j]-i[j-1]) 
    #print "temp_interval=",temp_interval
    #print "temp_interval",temp_interval
    if len(temp_interval) ==1:
        max_intv = temp_interval[0]
    else: 
        max_intv = k_means(temp_interval,'false', 'false')

    #max_intv=quantile(temp_interval,1.5)
    #print "max_intv=",max_intv
    #print temp_index                  
    temp_2=[]

    for i in temp_index:
        temp_2.append(i[0])
        if len(i) > 1:
            for j in range(1,len(i)):
                if i[j]-i[j-1] <= max_intv:
                    temp_2.append(i[j])
                else:
                    #if len(temp_2)>1:
                    chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
                    temp_2=[]
                    temp_2.append(i[j])
            #if len(temp_2)>1:
            chosn_index_finer.append(range(temp_2[0],temp_2[-1]+1))
            temp_2=[]
    #Not Finish
    last_tag = chosn_index_finer[0][-1]
    for i in chosn_index_finer:
        #before_len = len(i)
        tm = i[-1]
        if i[-1]!=len(dpot_L1)-1 and i[0]-last_tag>1:
            i.append(i[-1]+1)
        if i[0]!=0:
            i.append(i[0]-1)
        i.sort()
        last_tag = tm
    return chosn_index_finer
def partial_refinement_new(chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer_p=[]
    finer_r_p=[]
    finer_z_p=[]
    base_index = range(len(base))
    refined_base_index=[]
    print "len(base_index)=",len(base_index)
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)
    full_delta=range(finer_len)
    non_refine = list(set(full_delta).difference(set(chosn_index)))
    print "len(chosn_index)=",len(chosn_index)
    for i in range(len(chosn_index)):
        #if i%10000 ==0:
        #    print i
        index1 = chosn_index[i] // deci_ratio  
        index2 = chosn_index[i] % deci_ratio
        if index1 != len(base)-1:
            if index2!=0:
                finer[chosn_index[i]]=(base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i]
                finer_p.append((base[index1]+base[index1+1])*index2/deci_ratio + chosn_data[i])
                finer_r_p.append((base_r[index1]+base_r[index1+1])*index2/deci_ratio + chosn_r[i])
                finer_z_p.append((base_z[index1]+base_z[index1+1])*index2/deci_ratio + chosn_z[i])
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print finer[chosn_index[i]], data[chosn_index[i]] 
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
            else:
                finer[chosn_index[i]] = base[index1]
                finer_r[chosn_index[i]] = base_r[index1]
                finer_z[chosn_index[i]] = base_z[index1]
                finer_p.append(base[index1])
                finer_r_p.append(base_r[index1])
                finer_z_p.append(base_z[index1])
                refined_base_index.append(index1)
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print "ERROR2: index=",chosn_index[i]
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
        else:
            if index2!=0:
                finer[chosn_index[i]]= 2* base[index1]*index2/deci_ratio + chosn_data[i]
                finer_r[chosn_index[i]]=2 * base_r[index1]*index2/deci_ratio + chosn_r[i]
                finer_z[chosn_index[i]]=2 * base_z[index1]*index2/deci_ratio + chosn_z[i]
                finer_p.append(2* base[index1]*index2/deci_ratio + chosn_data[i])
                finer_r_p.append(2 * base_r[index1]*index2/deci_ratio + chosn_r[i])
                finer_z_p.append(2 * base_z[index1]*index2/deci_ratio + chosn_z[i])
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print "ERROR3: index=",chosn_index[i]
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
            else:
                finer[chosn_index[i]] = base[index1]
                finer_r[chosn_index[i]] = base_r[index1]
                finer_z[chosn_index[i]] = base_z[index1]
                finer_p.append(base[index1])
                finer_r_p.append(base_r[index1])
                finer_z_p.append(base_z[index1])
                refined_base_index.append(index1)                 
                #base_index.remove(index1)
                #if np.fabs(finer[chosn_index[i]] - data[chosn_index[i]])>0.000001:
                #    print "ERROR4: index=",chosn_index[i]
                #if np.fabs(finer_r[chosn_index[i]] - r[chosn_index[i]])>0.000001:
                #    print finer_r[chosn_index[i]], r[chosn_index[i]]
                #if np.fabs(finer_z[chosn_index[i]] - z[chosn_index[i]])>0.000001:
                #    print finer_z[chosn_index[i]], z[chosn_index[i]]
    for i in range(len(non_refine)):
        index1 = non_refine[i] // deci_ratio
        index2 = non_refine[i] % deci_ratio
        if index1 != len(base)-1:
            if index2!=0:
                finer[non_refine[i]]=(base[index1]+base[index1+1])*index2/deci_ratio
                finer_r[non_refine[i]]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio
                finer_z[non_refine[i]]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio
                #if (data[non_refine[i]] - finer[non_refine[i]] - delta[non_refine[i]])>0.000001:
                #    print "ERROR11:",data[non_refine[i]], finer[non_refine[i]], delta[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]] - delta_r[non_refine[i]])>0.000001:
                #    print "ERROR11:",r[non_refine[i]], finer_r[non_refine[i]], delta_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]] - delta_z[non_refine[i]])>0.000001:
                #    print "ERROR11:",z[non_refine[i]], finer_z[non_refine[i]], delta_z[non_refine[i]]

            else:
                finer[non_refine[i]] = base[index1]
                finer_r[non_refine[i]] = base_r[index1]
                finer_z[non_refine[i]] = base_z[index1]
                #if (data[non_refine[i]] - finer[non_refine[i]])>0.00001:
                #    print "ERROR22:",data[non_refine[i]], finer[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]])>0.00001:
                #    print "ERROR22:",r[non_refine[i]], finer_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]])>0.00001:
                #    print "ERROR22:",z[non_refine[i]], finer_z[non_refine[i]]
        else:
            if index2!=0:
                finer[non_refine[i]]= 2* base[index1]*index2/deci_ratio
                finer_r[non_refine[i]]=2* base_r[index1]*index2/deci_ratio
                finer_z[non_refine[i]]=2* base_z[index1]*index2/deci_ratio
                #if (data[non_refine[i]] - finer[non_refine[i]] - delta[non_refine[i]])>0.000001:
                #    print "ERROR33:",data[non_refine[i]], finer[non_refine[i]], delta[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]] - delta_r[non_refine[i]])>0.000001:
                #    print "ERROR33:",r[non_refine[i]], finer_r[non_refine[i]], delta_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]] - delta_z[non_refine[i]])>0.000001:
                #    print "ERROR33:",z[non_refine[i]], finer_z[non_refine[i]], delta_z[non_refine[i]]

            else:
                finer[non_refine[i]] = base[index1]
                finer_r[non_refine[i]] = base_r[index1]
                finer_z[non_refine[i]] = base_z[index1]
                #if (data[non_refine[i]] - finer[non_refine[i]])>0.00001:
                #    print "ERROR22:",data[non_refine[i]], finer[non_refine[i]]
                #if (r[non_refine[i]] - finer_r[non_refine[i]])>0.00001:
                #    print "ERROR22:",r[non_refine[i]], finer_r[non_refine[i]]
                #if (z[non_refine[i]] - finer_z[non_refine[i]])>0.00001:
                #    print "ERROR22:",z[non_refine[i]], finer_z[non_refine[i]]
    remain_base_index = list(set(base_index).difference(set(refined_base_index)))
    for i in remain_base_index:
        finer_p.append(base[i])
        finer_r_p.append(base_r[i])
        finer_z_p.append(base_z[i])
    return finer, finer_r, finer_z,finer_p, finer_r_p, finer_z_p
          
def partial_refinement(chosn_index, finer_len, chosn_data,chosn_r,chosn_z, base, base_r, base_z, deci_ratio):
    finer=np.zeros(finer_len, dtype = np.float64)
    finer_r=np.zeros(finer_len, dtype = np.float64)
    finer_z=np.zeros(finer_len, dtype = np.float64)
    start = 0
    inc = 0
    i=0
    num_refine=0
    refine_index=[]
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
                    finer[n]=(base[index1]+base[index1+1])*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=(base_r[index1]+base_r[index1+1])*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=(base_z[index1]+base_z[index1+1])*index2/deci_ratio+chosn_z[inc]
                    refine_index.append(n)
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    refine_index.append(n)
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            else:
                if index2!=0:
                    finer[n]=2*base[index1]*index2/deci_ratio+chosn_data[inc]
                    finer_r[n]=2*base_r[index1]*index2/deci_ratio+chosn_r[inc]
                    finer_z[n]=2*base_z[index1]*index2/deci_ratio+chosn_z[inc]
                    refine_index.append(n)
                    num_refine+=1
                else:
                    finer[n]=base[index1]
                    finer_r[n]=base_r[index1]
                    finer_z[n]=base_z[index1]
                    refine_index.append(n)
                    #finer_p.append(base[index1])
                    #finer_r_p.append(base_r[index1])
                    #finer_z_p.append(base_z[index1])
            inc+=1
        start = chosn_index[i+1]+1
        if i == len(chosn_index)-2:
            for j in range(start,finer_len):
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
    #print refine_index    
    #print num_refine
    #print "percentage of selected points= ", num_refine/(finer_len-len(base))

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
    #print source
    y=np.array(source).reshape(-1,1)
        
    km=KMeans(n_clusters=2)
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
    #group_index=[]

    #for i in range(len(km_label)):
    #    if km_label[i] in group: group_index.append(i)
    limit = 0
    for i in range(len(source)):
        if km_label[i] in group:
            #print km_label[i]
            if source[i]>limit:
                #print source[i],limit
                limit = source[i]
    return limit

def noise_threshold(noise, peak_noise, no_noise):
    print "Actual peak_noise=", peak_noise
    print "Actual no_noise=", no_noise
    #peak_noise = 40
    #no_noise = 90
    #print "peak_noise=", peak_noise
    #print "no_noise=", no_noise
    k=(1.0-0.0)/(no_noise - peak_noise)
    b =1.0 - k * no_noise

    if noise < peak_noise:
        threshold = 0.0
    elif noise > no_noise:
        threshold = 1.0 
    else:
        threshold = noise*k+b
        #thre.append(threshold)

    return threshold

def FFT (Fs,data):
    L = len (data)
    N =int(np.power(2,np.ceil(np.log2(L))))
    FFT_y = np.abs(fft(data,N))/L*2 
    Fre = np.arange(int(N/2))*Fs/N
    FFT_y = FFT_y[range(int(N/2))]
    return Fre, FFT_y

def prediction_noise_wave(samples, hi_freq_ratio,timestep_interval):

    sample_rate = 1/timestep_interval
    Nsamples = len(samples)
    print "sample rate = ",sample_rate
    #amp = fft.fft(samples)/(Nsamples/2.0)
    amp = fft.fft(samples)/Nsamples
    #amp_complex_h = amp[range(int(len(samples)/2))]
    amp_complex_h = amp
    amp_h = np.absolute(amp_complex_h)

    freq=fft.fftfreq(amp.size,1/sample_rate)
    freq_h = freq
    #freq_h = freq[range(int(len(samples)/2))] 

    if amp_h[0]>1e-10:
        threshold = np.max(np.delete(amp_h,0,axis=0))*hi_freq_ratio
        dc = amp_h[0]
        start_index = 1
    else:
        threshold = np.max(amp_h)*hi_freq_ratio
        dc = 0.0
        start_index = 0
    #print "dc",dc
    #print "threshold",threshold
    selected_freq = []
    selected_amp = []
    selected_complex=[]
    for i in range(start_index,len(amp_h)):
        if amp_h[i]>=threshold:
            selected_freq.append(freq_h[i])
            selected_amp.append(amp_h[i])
            selected_complex.append(amp_complex_h[i])

    selected_phase = np.arctan2(np.array(selected_complex).imag,np.array(selected_complex).real)

    for i in range(len(selected_phase)):
        if np.fabs(selected_phase[i])<1e-10:
            selected_phase[i]=0.0
    #print "future_timestep", future_timestep
    #future_timestep=np.array([0])
    return dc, selected_amp, selected_freq, selected_phase   

 
def get_prediction_threshold(dc, selected_amp, selected_freq, selected_phase, time, peak_noise, no_noise):
    sig = dc
    for i in range(len(selected_freq)):
        sig += selected_amp[i]*np.cos(2*np.pi*selected_freq[i]*time+ selected_phase[i])
    if sig < 0 :
        sig = 1.1
    print "Noise amplitude=",sig
    threshold = noise_threshold(sig, peak_noise, no_noise)
    return threshold, sig

def get_chosn_data_index(threshold):
    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_compressed=f.read(4325048*8)
    dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    #r_L1_compressed=f.read(2975952*8)
    #r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    #z_L1_compressed=f.read(2516984*8)
    #z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close()
     
    if threshold == 0.0:
        chosn_index_L1 = []            
    elif threshold == 1.0:
        chosn_index_L1 = [range(len(dpot_L1))]
    else:
        chosn_index_L1 = find_augment_points_gradient(dpot_L1,[range(len(dpot_L1))], 1-threshold)

    return chosn_index_L1    

def calc_area(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1,p2,p3
    #return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)
    area = abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))*0.5)
    return area

def high_potential_area(dpot,R,Z,thre):
    start = time.time()
    points = np.transpose(np.array([Z,R]))
    Delaunay_t = Delaunay(points)
    conn=Delaunay_t.simplices
    area=0.0
    for i in range(len(conn)):
        index1=conn[i][0]
        index2=conn[i][1]
        index3=conn[i][2]
        #if (dpot[index1]>thre and dpot[index2]>thre) or (dpot[index1]>thre and dpot[index3]>thre) or (dpot[index2]>thre and dpot[index3]>thre):
        if (dpot[index1]+dpot[index2]+dpot[index3])/3.0 > thre:
            each_area=calc_area((R[index1],Z[index1]),(R[index2],Z[index2]),(R[index3],Z[index3]))
            area = area + each_area
    end = time.time()
    print "High potential analysis time = ", end - start
    return area

def fully_refine(deci_ratio,timestep,tag,time_tail):
    a = time.time()
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
    #   print "Connected to the cluster."

    #if not cluster.pool_exists('tier0_pool'):
	#    raise RuntimeError('No data pool exists')
    #ioctx_0 = cluster.open_ioctx('tier0_pool')
    start_ssd = time.time()

    reduced_len = 4876
    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_str=f.read(reduced_len*8)
    #dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    r_L1_str=f.read(reduced_len*8)
    #r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    z_L1_str=f.read(reduced_len*8)
    #z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close()
    end_ssd = time.time()
    dpot_L1=struct.unpack(str(reduced_len)+'d',dpot_L1_str)
    r_L1=struct.unpack(str(reduced_len)+'d',r_L1_str)
    z_L1=struct.unpack(str(reduced_len)+'d',z_L1_str)

    read_ssd_time = end_ssd- start_ssd
    print "Time for reading reduced data from SSD =", read_ssd_time
    #dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
    #r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0],0)
    #z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0],0)
    #dpot_L1=zfpy._decompress(dpot_L1_str, 4, [10556545], tolerance=0.01)
    #r_L1=zfpy._decompress(r_L1_str, 4, [10556545], tolerance=0.01)
    #z_L1=zfpy._decompress(z_L1_str, 4, [10556545], tolerance=0.01)
    #tag=str(int(ioctx_0.stat("dpot_L1")[0]/8))
    #dpot_L1=struct.unpack(tag+'d',dpot_L1_str)
    #r_L1=struct.unpack(tag+'d',r_L1_str)
    #z_L1=struct.unpack(tag+'d',z_L1_str)
    #ioctx_0.close()
    if timestep == 0:
        ssd_time=[]
        ssd_time.append(read_ssd_time)
        print "Total ssd_time = ",  ssd_time
        np.savez("ssd_time.npz", ssd_time = np.array(ssd_time))
    else:
        f_ssd = np.load("ssd_time.npz")
        ssd_time=f_ssd["ssd_time"]
        ssd_time = ssd_time.tolist()
        ssd_time.append(read_ssd_time)
        print "Total ssd_time = ",  ssd_time
        np.savez("ssd_time.npz", ssd_time = np.array(ssd_time))
  

    if not cluster.pool_exists('tier2_pool'):
	    raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')
    #delta_L0_L1_o_str = ioctx_2.read("delta_L0_L1_o",ioctx_2.stat("delta_L0_L1_o")[0],0)
    #delta_L0_L1_o = struct.unpack(str(int(ioctx_2.stat("delta_L0_L1_o")[0]/8))+'d',delta_L0_L1_o_str)
    start=time.time()
    if timestep !=-1:
        print "read all\n"
        aaa = time.time()
        delta_L0_L1_str = ioctx_2.read("delta_L0_L1_o",ioctx_2.stat("delta_L0_L1_o")[0],0)
        bbb = time.time()
        print "bw = ",ioctx_2.stat("delta_L0_L1_o")[0]/1024.0/1024.0/(bbb-aaa)
    #print "delta_L0_L1 size =", sys.getsizeof(delta_L0_L1)/1024/1024
        delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1_o",ioctx_2.stat("delta_r_L0_L1_o")[0],0)
        delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1_o",ioctx_2.stat("delta_z_L0_L1_o")[0],0)
    else:
        print "read 8bytes"
        delta_L0_L1_str = ioctx_2.read("delta_L0_L1_o",8,0)
        delta_r_L0_L1_str= ioctx_2.read("delta_r_L0_L1_o",8,0)
        delta_z_L0_L1_str = ioctx_2.read("delta_z_L0_L1_o",8,0)

    end = time.time()
    delta_read_time = end-start
    print "Data read = %d Mb"%int((ioctx_2.stat("delta_L0_L1_o")[0]+ioctx_2.stat("delta_r_L0_L1_o")[0]+ioctx_2.stat("delta_z_L0_L1_o")[0])/1024/1024)
    print "Delta reading time = ", delta_read_time
    #delta_L0_L1 = zfpy._decompress(delta_L0_L1_str, 4, [4992221], tolerance=0.01)
    #delta_r_L0_L1 = zfpy._decompress(delta_r_L0_L1_str, 4, [4992221], tolerance=0.01)
    #delta_z_L0_L1 = zfpy._decompress(delta_z_L0_L1_str, 4, [4992221], tolerance=0.01)
    if timestep !=-1:
        delta_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_L0_L1_o")[0]/8))+'d',delta_L0_L1_str)
        delta_r_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_r_L0_L1_o")[0]/8))+'d',delta_r_L0_L1_str)
        delta_z_L0_L1 = struct.unpack(str(int(ioctx_2.stat("delta_z_L0_L1_o")[0]/8))+'d',delta_z_L0_L1_str)
        sample_bandwidth = (ioctx_2.stat("delta_L0_L1_o")[0]+ ioctx_2.stat("delta_r_L0_L1_o")[0] + ioctx_2.stat("delta_z_L0_L1_o")[0])/1024/1024/delta_read_time
    else:
        delta_L0_L1 = struct.unpack('d',delta_L0_L1_str)
        delta_r_L0_L1 = struct.unpack('d',delta_r_L0_L1_str)
        delta_z_L0_L1 = struct.unpack('d',delta_z_L0_L1_str)
        sample_bandwidth = (8*3)/1024/1024/delta_read_time
    #print "Delta reading bandwidth = ", (ioctx_2.stat("delta_L0_L1")[0]+ ioctx_2.stat("delta_r_L0_L1")[0] + ioctx_2.stat("delta_z_L0_L1")[0])/1024/1024/delta_time
    fname = "sample_"+str(tag)+".npz"
        
    if timestep != 0:
        fp = np.load(fname)
        sample_bd = fp['sample_bandwidth']
        sample_read_time = fp['sample_read_time']
    else:
        sample_bd = np.array([]) 
        sample_read_time = np.array([])
    sample_bd = sample_bd.tolist()
    sample_read_time = sample_read_time.tolist()
    
    sample_bd.append(sample_bandwidth)
    sample_read_time.append(delta_read_time)
    if timestep == time_tail:
    #if len(sample_bd) == update_interval:
        fname1 = "sample_"+str(tag+1)+".npz"
        print "Written to ", fname1
        np.savez(fname1, sample_bandwidth = np.array([sample_bandwidth]),sample_read_time = np.array([delta_read_time]))
    bdstr=''
    for i in range(len(sample_bd)-1):
        bdstr += str(sample_bd[i])+","
    bdstr += str(sample_bd[-1])
    print "Total bandwidth samples=",bdstr
    timestr=''
    for i in range(len(sample_read_time)-1):
        timestr += str(sample_read_time[i])+","
    timestr += str(sample_read_time[-1])
    print "Total time samples=",timestr
    np.savez(fname, sample_bandwidth = np.array(sample_bd), sample_read_time = np.array(sample_read_time))
    #tag = str(int(ioctx_2.stat("delta_L0_L1")[0]/8))
    #delta_L0_L1 = struct.unpack(tag+'d',delta_L0_L1_str)
    #print "delta_L0_L1 size =", sys.getsizeof(delta_L0_L1)/1024/1024

    #delta_r_L0_L1 = struct.unpack(tag+'d',delta_r_L0_L1_str)
    #delta_z_L0_L1 = struct.unpack(tag+'d',delta_z_L0_L1_str)
    #aa=time.time()
    #w_finer, w_finer_r,w_finer_z = whole_refine(dpot_L1,r_L1,z_L1,delta_L0_L1,delta_r_L0_L1,delta_z_L0_L1,deci_ratio)
    #bb=time.time()
    #finer, finer_r,finer_z,finer_p, finer_r_p, finer_z_p = partial_refinement_new(range(len(delta_L0_L1)), len(delta_L0_L1), delta_L0_L1, delta_r_L0_L1, delta_z_L0_L1, dpot_L1, r_L1, z_L1, deci_ratio)
    #refine_start = time.time()
    finer, finer_r,finer_z = partial_refinement((0,len(delta_L0_L1)-1), len(delta_L0_L1), delta_L0_L1, delta_r_L0_L1, delta_z_L0_L1, dpot_L1, r_L1, z_L1, deci_ratio)
    #refine_end = time.time()
    #print "Refinement time = ", refine_end- refine_start
    ioctx_2.close()
    cluster.shutdown()
    #print "start plot\n"
    #a=time.time()
    b=time.time()
    print "Fully refinement function time = ", b-a
    #high_p_area = high_potential_area(finer, finer_r,finer_z,thre)
    #return high_p_area

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
    #plt.triplot(r,z,conn)
    plt.tricontourf(r, z, conn, data,cmap=plt.cm.jet, levels=np.linspace(np.min(data),np.max(data),num=25));
    #plt.colorbar();
    plt.xticks([])
    plt.yticks([])
    for key, spine in ax.spines.items():
        # 'left', 'right', 'bottom', 'top'
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    plt.savefig(filename, format='png')

def partial_refine(deci_ratio,timestep,psnr,ctag,time_tail, peak_noise, no_noise, time_interval):
#def partial_refine(deci_ratio,timestep,psnr,ctag,update_interval,thre, peak_noise, no_noise, time_interval,original_data, original_r,original_z,delta,delta_r,delta_z):
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
    #    print "Connected to the cluster."

    if not cluster.pool_exists('tier2_pool'):
        raise RuntimeError('No data pool exists')
    ioctx_2 = cluster.open_ioctx('tier2_pool')

    #dpot_L1_str = ioctx_0.read("dpot_L1",ioctx_0.stat("dpot_L1")[0],0)
    #r_L1_str = ioctx_0.read("r_L1",ioctx_0.stat("r_L1")[0],0)
    #z_L1_str = ioctx_0.read("z_L1",ioctx_0.stat("z_L1")[0],0)
    #dpot_L1=zfpy._decompress(dpot_L1_str, 4, [2496111], tolerance=0.01)
    #r_L1=zfpy._decompress(r_L1_str, 4, [2496111], tolerance=0.01)
    #z_L1=zfpy._decompress(z_L1_str, 4, [2496111], tolerance=0.01)
    #tag=str(int(ioctx_0.stat("dpot_L1")[0]/8))
    #dpot_L1=struct.unpack(tag+'d',dpot_L1_str)
    #r_L1=struct.unpack(tag+'d',r_L1_str)
    #z_L1=struct.unpack(tag+'d',z_L1_str)
    #sa = time.time()
    
    start_ssd = time.time()

    reduced_len = 4876
    filename = "reduced_data.bin"
    f = open(filename, "rb")
    dpot_L1_str=f.read(reduced_len*8)
    #dpot_L1=zfpy._decompress(dpot_L1_compressed, 4, [2496111], tolerance=0.01)
    r_L1_str=f.read(reduced_len*8)
    #r_L1=zfpy._decompress(r_L1_compressed, 4, [2496111], tolerance=0.01)
    z_L1_str=f.read(reduced_len*8)
    #z_L1=zfpy._decompress(z_L1_compressed, 4, [2496111], tolerance=0.01)
    f.close()
    end_ssd = time.time()
    dpot_L1=struct.unpack(str(reduced_len)+'d',dpot_L1_str)
    r_L1=struct.unpack(str(reduced_len)+'d',r_L1_str)
    z_L1=struct.unpack(str(reduced_len)+'d',z_L1_str)

    read_ssd_time = end_ssd- start_ssd
    print "Time for reading reduced data from SSD =", read_ssd_time

    f_ssd = np.load("ssd_time.npz") 
    ssd_time=f_ssd["ssd_time"]
    ssd_time = ssd_time.tolist()
    ssd_time.append(read_ssd_time)
    print "Total ssd_time = ",  ssd_time
    np.savez("ssd_time.npz", ssd_time = np.array(ssd_time))
    fname = "recontructed_noise_"+str(ctag-1)+".npz"
    f_np = np.load(fname)
    dc = f_np["recontructed_noise_dc"]
    s_freq = f_np["recontructed_noise_freq"]
    s_amp = f_np["recontructed_noise_amp"]
    s_phase = f_np["recontructed_noise_phase"]
    peak_noise_a = f_np["peak_noise"]
    no_noise_a = f_np["no_noise"]
    print "Detected peak_noise", peak_noise_a
    print "Detected no_noise", no_noise_a
    if peak_noise < peak_noise_a or no_noise > no_noise_a:
        print "Peak/no noise setting error!\n"
    #peak_noise = int((peak_noise_a//10+1)*10)
    #no_noise = int((no_noise_a//10)*10)
    thre, sig = get_prediction_threshold(dc, s_amp, s_freq, s_phase, timestep, peak_noise, no_noise)
    
    print "threshold=",thre

    #thre = 0.891035875465176064
    #sig = 70.46215252791056
    #thre = 0.99
    #chosn_index = get_chosn_data_index(thre)
    
    #chosn_index_e = []
    #for i in range(len(chosn_index)):
    #    chosn_index_e.append(chosn_index[i][0]*deci_ratio)
    #    chosn_index_e.append(chosn_index[i][-1]*deci_ratio)
    #chosn_data_str = ""
    #chosn_r_str = ""
    #chosn_z_str = ""
    #chosn_len = 0
    #delta_read_time = 0.0
    #i=0
    #segment_num = 0
    #segment_size = []
    ##print "chosn_index_e",chosn_index_e
    #while(i<len(chosn_index_e)):
    #    start = time.time()
    #    temp_str = ioctx_2.read("delta_L0_L1_o",(chosn_index_e[i+1]-chosn_index_e[i]+1)*8,chosn_index_e[i]*8)
    #    temp_r_str = ioctx_2.read("delta_r_L0_L1_o",(chosn_index_e[i+1]-chosn_index_e[i]+1)*8,chosn_index_e[i]*8)
    #    temp_z_str = ioctx_2.read("delta_z_L0_L1_o",(chosn_index_e[i+1]-chosn_index_e[i]+1)*8,chosn_index_e[i]*8) 
    #    end = time.time()
    #    delta_read_time += end-start
    #    #check overflow
    #    chosn_data_str = chosn_data_str + temp_str
    #    chosn_r_str =  chosn_r_str + temp_r_str
    #    chosn_z_str =  chosn_z_str + temp_z_str 
    #    chosn_len+= chosn_index_e[i+1]-chosn_index_e[i]+1
    #    segment_size.append((chosn_index_e[i+1]-chosn_index_e[i]+1)*8+(chosn_index_e[i+1]-chosn_index_e[i]+1)*8+(chosn_index_e[i+1]-chosn_index_e[i]+1)*8)
    #    segment_num += 1	
    #    i+=2
    #chosn_data = struct.unpack(str(chosn_len)+'d',chosn_data_str)
    #chosn_r = struct.unpack(str(chosn_len)+'d',chosn_r_str)
    #chosn_z = struct.unpack(str(chosn_len)+'d',chosn_z_str)
    #interval = 100
    #percentage = np.arange(0,1000,interval)/1000
    number_blocks = 50
    block_interval = 1/number_blocks
    if thre % block_interval > block_interval/2:
        block_num = thre // block_interval+1
    else:
        block_num = thre // block_interval
    #if sig < 20:
    #    block_num = block_num //2
    #elif sig >= 20 and sig < 40:
    #    block_num = int(block_num /2.5)
    #elif sig >= 40 and sig < 60: 
    #    block_num = int(block_num /3)
    #elif sig >=40 and sig < 90:
    #    block_num = int(block_num /3.5)
    
    if block_num ==0:
        block_num = 1
        print "Noise is too large, threshold is less than 0.05, set block_num =1\n"
    print "Number of blocks =", block_num
    delta_index_L0_L1_str = ""
    delta_L0_L1_str = ""
    delta_r_L0_L1_str = ""
    delta_z_L0_L1_str = ""
    delta_len=0
    delta_read_time = 0.0
    finer_len = int(ioctx_2.stat("delta_L0_L1_o")[0]/8)
    if sig > peak_noise:
        for i in range(int(block_num)):
            #print "block tag=",i*block_interval
			#print i*0.1
            chosn_delta_name= "delta_L0_L1_"+str(i*block_interval)
            chosn_delta_r_name = "delta_r_L0_L1_"+str(i*block_interval)
            chosn_delta_z_name = "delta_z_L0_L1_"+str(i*block_interval)
            chosn_index_name = "delta_index_" + str(i*block_interval)
			#print chosn_delta_name, chosn_delta_r_name, chosn_delta_z_name, chosn_index_name
            start = time.time()
            #if block_num != 1:
            delta_index_L0_L1_str += ioctx_2.read(chosn_index_name,ioctx_2.stat(chosn_index_name)[0],0)
            delta_L0_L1_str += ioctx_2.read(chosn_delta_name,ioctx_2.stat(chosn_delta_name)[0],0)
            delta_r_L0_L1_str += ioctx_2.read(chosn_delta_r_name,ioctx_2.stat(chosn_delta_r_name)[0],0) 
            delta_z_L0_L1_str += ioctx_2.read(chosn_delta_z_name,ioctx_2.stat(chosn_delta_z_name)[0],0)		
            delta_len += int(ioctx_2.stat(chosn_delta_name)[0]/8)
        	#else:
        	#	delta_index_L0_L1_str += ioctx_2.read(chosn_index_name,4,0)
        	#	delta_L0_L1_str += ioctx_2.read(chosn_delta_name,8,0)
        	#	delta_r_L0_L1_str += ioctx_2.read(chosn_delta_r_name,8,0)
        	#	delta_z_L0_L1_str += ioctx_2.read(chosn_delta_z_name,8,0)
        	#	delta_len += 1            
            end = time.time()
			#print ioctx_2.stat(chosn_index_name)[0]/4, ioctx_2.stat(chosn_delta_name)[0]/8, ioctx_2.stat(chosn_delta_r_name)[0]/8, ioctx_2.stat(chosn_delta_z_name)[0]/8
            delta_read_time+= end-start
			#delta_len += int(ioctx_2.stat(chosn_delta_name)[0]/8) 

        delta_index = struct.unpack(str(delta_len)+'i',delta_index_L0_L1_str)
        delta_L0_L1 = struct.unpack(str(delta_len)+'d',delta_L0_L1_str)
        delta_r_L0_L1 = struct.unpack(str(delta_len)+'d',delta_r_L0_L1_str)
        delta_z_L0_L1 = struct.unpack(str(delta_len)+'d',delta_z_L0_L1_str)
	   
        print "Read chosn delta time=",delta_read_time
		#print "Number of segment = ",segment_num
		#print "All segment size = ",segment_size
        print "Number of selected element=",delta_len
        sample_bandwidth = (delta_len*3*8+delta_len*4)/1024/1024/delta_read_time
        #finer_len = int(ioctx_2.stat("delta_L0_L1_o")[0]/8)
        print "Delta fetching percentage=",delta_len*100/finer_len
    else:
        sample_bandwidth = sig
        print "Not read any delta!\n"
    #print "finer_len=",finer_len
    ioctx_2.close()
    cluster.shutdown()
    fname = "sample_"+str(ctag)+".npz"
    if timestep == 0:
    	print "Error, time step for partial refinement couldn't be 0!\n"
    
    fp = np.load(fname)
    sample_bd = fp['sample_bandwidth']
    sample_read_time = fp['sample_read_time']
    sample_bd = sample_bd.tolist()
    sample_read_time = sample_read_time.tolist()

    sample_bd.append(sample_bandwidth)
    sample_read_time.append(delta_read_time)
    if timestep == time_tail:
    #if len(sample_bd) == update_interval:
        fname1 = "sample_"+str(ctag+1)+".npz"
        print "Written to ", fname1
        np.savez(fname1, sample_bandwidth = np.array([sample_bandwidth]),sample_read_time = np.array([delta_read_time]))
    bdstr=''
    for i in range(len(sample_bd)-1):
        bdstr += str(sample_bd[i])+","
    bdstr += str(sample_bd[-1])
    print "Total bandwidth samples=",bdstr
    timestr=''
    for i in range(len(sample_read_time)-1):
        timestr += str(sample_read_time[i])+","
    timestr += str(sample_read_time[-1])
    print "Total time samples=",timestr
    np.savez(fname, sample_bandwidth = np.array(sample_bd), sample_read_time = np.array(sample_read_time))
    #if 1==0:
    if sig > peak_noise:
        finer, finer_r,finer_z, finer_p, finer_r_p,finer_z_p = partial_refinement_new(delta_index, finer_len, delta_L0_L1, delta_r_L0_L1, delta_z_L0_L1, dpot_L1, r_L1, z_L1, deci_ratio)
    else:
        finer = dpot_L1
        finer_r = r_L1
        finer_z = z_L1
        finer_p = dpot_L1
        finer_r_p = r_L1
        finer_z_p = z_L1        
    finer_name = "xgc/dif_bandwidth_low_high/xgc_"+str(peak_noise)+"_"+str(no_noise)+"_"+str(timestep)+"_"+str(ctag)+".npz"
    np.savez(finer_name, finer = finer_p, finer_r = finer_r_p, finer_z = finer_z_p, psnr_finer = finer, psnr_finer_r = finer_r, psnr_finer_z = finer_z)
    #bb =time.time()
    #print "Time for one time partial refinement=",bb-aa
    #high_p_area = high_potential_area(finer, finer_r,finer_z,thre)
    #plot(finer_p,finer_r_p,finer_z_p,str(int(block_num))+"_123.png") 
    if psnr =="True":
        psnr_start = time.time()
        filename = "full_data.bin"
        f = open(filename, "rb")
        dpot_str = f.read(finer_len*8)
        r_str = f.read(finer_len*8)
        z_str = f.read(finer_len*8)
        f.close()
        number_of_original_elements = str(finer_len)
        dpot=struct.unpack(number_of_original_elements+'d',dpot_str)
        r=struct.unpack(number_of_original_elements+'d',r_str)
        z=struct.unpack(number_of_original_elements+'d',z_str)
        #for i in delta_index:
        #for i in range(finer_len):
        #    if np.fabs(finer[i] - dpot[i]) >0.000001:
        #        print "finer[%d]=%f, dpot[%d]=%f\n"%(i, finer[i], i, dpot[i])
        #    if np.fabs(finer_r[i] - r[i]) >0.000001:
        #        print "finer_r[%d]=%f, r[%d]=%f\n"%(i, finer_r[i], i, r[i])
        #    if np.fabs(finer_z[i] - z[i])>0.000001:
        #        print "finer_z[%d]=%f, z[%d]=%f\n"%(i, finer_z[i], i, z[i])
        data_len=[len(dpot_L1),len(dpot)]
        print data_len
        #psnr_original=psnr_c(dpot,dpot_L1,data_len,deci_ratio, 1)
        if len(finer)!=len(dpot):
            print "finer len error!\n"
            print "len(finer)=%d len(dpot)=%d\n"%(len(finer), len(dpot))
        #for i in range(len(finer)):
        #    if finer[i] - dpot[i]>0.01:
        #        print "finer[i]=%f, dpot[i]=%f\n"%(finer[i], dpot[i])
        if sig > peak_noise:
            psnr_finer=psnr_c(dpot, finer, data_len, deci_ratio, 0)
        else:
            psnr_finer = psnr_c(dpot,dpot_L1,data_len,deci_ratio, 1)
        print "finer PSNR=",psnr_finer
        #print "original PSNR=",psnr_original
        if timestep == time_interval:
            np.savez("psnr.npz", psnr = [psnr_finer])
            print "Total PSNR =", [psnr_finer]
        else:
            fpp = np.load("psnr.npz")
            s_psnr = fpp["psnr"]
            s_psnr = s_psnr.tolist()
            s_psnr.append(psnr_finer)
            np.savez("psnr.npz", psnr = s_psnr)
            print "Total PSNR =", s_psnr
        psnr_end = time.time()
        print "Time for calculate PSNR = ",psnr_end-psnr_start
    #return high_p_area

def psnr_c(original_data, base, leveldata_len, deci_ratio, level_id):
    for i in range(len(leveldata_len)-level_id,len(leveldata_len)):
        leveldata=np.zeros(leveldata_len[i])
        for j in range(leveldata_len[i]):
            index1=j//deci_ratio
            index2=j%deci_ratio
            if index1!= len(base)-1:
                if index2 != 0:
                    leveldata[j]=(base[index1]+base[index1+1])*index2/deci_ratio
                else:
                    leveldata[j]=base[index1]
            else:
                if index2 != 0:
                    leveldata[j]=(base[index1]*2)*index2/deci_ratio
                else:
                    leveldata[j]=base[index1]
        base=leveldata
    if len(base) !=len(original_data):
        print "len(leveldata) !=len(original_data)"

    MSE = 0.0
    for i in range(len(original_data)):
        #print i, original_data[i]-base[i]
        MSE=(original_data[i]-base[i])*(original_data[i]-base[i])+MSE
    MSE=MSE/len(original_data)
    if MSE < 1e-6:
        MSE=0.0
    if MSE ==0.0:
        print "Couldn't get PSNR of two identical data."
        return 0
    else:
        psnr=10*math.log(np.max(original_data)**2/MSE,10)
    #print "psnr=",psnr
    return psnr
 
def work_flow(deci_ratio, time_interval, psnr,peak_noise, no_noise):
    tag = 0
    timestep = 0
    last_tag = 0
    t_sample = np.arange(time_interval,1201,time_interval)
    t_tail = t_sample[-1]
    t_tail_whole = t_sample[-1]
    
    for i in range(0,3000,time_interval):
        start=time.time()
        print "time = %d\n" %(i)
        #if int(i/time_interval) %(update_interval-1) == 1 and i !=time_interval and i !=2385:
        if i == t_tail_whole+time_interval:
            print "start updating prediction"
            a=time.time()
            fname="sample_"+str(tag)+".npz"
            fp = np.load(fname)
            samples_bd = fp['sample_bandwidth']
            dc, s_amp, s_freq, s_phase = prediction_noise_wave(samples_bd, 0.5, time_interval)
            t = np.arange(i-1200*(tag+1),1200+1, time_interval)
            t_tail = t[-1]
            t_tail_whole = t[-1] + 1200*(tag+1)   
            sig = dc  
            for j in range(len(s_freq)):
                sig += s_amp[j]*np.cos(2*np.pi*s_freq[j]*t+ s_phase[j])
            print "Predicted sig bandwidth=", sig.tolist()
            print "Predicted time steps = ", t.tolist()
            print "Predicted sig bandwidth Max = %f, Min = %f"%(np.max(sig), np.min(sig))
            noise_low_bar = 0.0
            for m in range(len(t)):
                if t[m]%60 ==0 or t[m]%80 ==0 or t[m]%100 ==0:
                    if sig[m]>noise_low_bar:
                        noise_low_bar = sig[m]
            
            r_noise_name = "recontructed_noise_"+str(tag)+".npz"
            np.savez(r_noise_name, recontructed_noise_dc = dc, recontructed_noise_amp = s_amp,recontructed_noise_freq = s_freq, recontructed_noise_phase = s_phase, no_noise = max(sig), peak_noise = noise_low_bar)
            tag += 1
            b=time.time()
            print "Updating prediction time = ", b-a
        if tag == 0:
            print "fully refinement\n"
            timestep = i
            print timestep
            fully_refine(deci_ratio,i, tag, t_tail)
        else:
            print i,  time_interval
            timestep = i-1200*tag
            #timestep = i % ((update_interval-1)*time_interval) 
            print "timestep=",timestep
            print "partial refinement\n"
            print timestep
            partial_refine(deci_ratio, timestep, psnr, tag,t_tail, peak_noise, no_noise, time_interval)         
        end = time.time()
         
        #timestep += time_interval
        print "Analysis time = ",end-start
        time.sleep(time_interval-(end-start)) 
        #e=time.time()
        #print "One time workflow start from %d = %f\n"%(i*time_interval,e-a)


deci_ratio=1024
time_interval = 25
frequency_cut_off = 0.5
peak_noise =60
no_noise = 120

work_flow(deci_ratio,time_interval,"false", peak_noise, no_noise)
