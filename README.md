# Application level QoS control
This repository contains materials to reproduce the results for SC 2020 paper "Taming I/O Variation on QoS-less HPC Storage: What Can Applications do?"

## Prerequisite

1. We evaluate our scheme using CEPH file system. So firstly you need to set up the cluster based on CEPH. We use [chameleon](https://www.chameleoncloud.org/) as our testbed. We configure our shared storage system with two storage nodes. Please refer to the [CEPH documents](https://docs.ceph.com/docs/master/install/ceph-deploy/) to do so. 
2. Since we utilize the ADIOS to refactor and write data to CEPH file system. Once you set up the cluster, you need to build and install the ADIOS in each client node. Please refer to the [ADIOS documentation](https://www.olcf.ornl.gov/center-projects/adios/). 

## How to use the code

1. Please place the full data including raw data files and mesh files under "ADIOS/examples/C/global-array/larger_data", change the related directory in "test_xgc.c" and set up the decimation ratio in "test_xgc.xml"
2. Once you run "test_xgc" to refactor the original data and write them to the storage pool, run "write_block_delta.py" to divide the delta into 50 blocks based on the gradient of reduced data in local storage.
3. Run "qos_control.py" to achieve the QoS control for the application which can manage the I/O variation and increase the I/O performance. This will generate the refined data for each timestep based on the estimated noises in shared storage.
4. The analysis of our evaluation is offline, "xgc_analysis.py", "astro2d_analysis.py" and "cfd_analysis.py" are used for the data analysis.
5. When testing the load balancing, you can run "sampling.py" to repeatedly read one object, and collect the perceived bandwidth.
6. You can run "./interference [checkpoint size (MB)] [checkpoint interval (secs)]" in different client nodes to generate the periodic interferences.

## Note

Remember to set the correct parameters in the scripts such as decimation ratio, analysis time interval, reduced data length to make sure they can be ran correctly
