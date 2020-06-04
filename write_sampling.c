#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rados/librados.h>
#include <time.h>
#include <sys/time.h>
int main (int argc, const char **argv)
{

        /* Declare the cluster handle and required arguments. */
        rados_t cluster;
        char cluster_name[] = "ceph";
        char user_name[] = "client.admin";
        uint64_t flags = 0;
	    double * data=0, * data1=0;
        int datasize=0;

        /* Initialize the cluster handle with the "ceph" cluster name and the "client.admin" user */
        int err;
        err = rados_create2(&cluster, cluster_name, user_name, flags);

        if (err < 0) {
                fprintf(stderr, "%s: Couldn't create the cluster handle! %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nCreated a cluster handle.\n");
        }


        /* Read a Ceph configuration file to configure the cluster handle. */
        err = rados_conf_read_file(cluster, "/etc/ceph/ceph.conf");
        if (err < 0) {
                fprintf(stderr, "%s: cannot read config file: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nRead the config file.\n");
        }

        /* Read command line arguments */
        /*err = rados_conf_parse_argv(cluster, argc, argv);
        if (err < 0) {
                fprintf(stderr, "%s: cannot parse command line arguments: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nRead the command line arguments.\n");
	*/
	
	/* Connect to the cluster */
        err = rados_connect(cluster);
        if (err < 0) {
                fprintf(stderr, "%s: cannot connect to cluster: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nConnected to the cluster.\n");
        }

	rados_ioctx_t io;
        char *poolname = "tier2_pool";

        err = rados_ioctx_create(cluster, poolname, &io);
        if (err < 0) {
                fprintf(stderr, "%s: cannot open rados pool %s: %s\n", argv[0], poolname, strerror(-err));
                rados_shutdown(cluster);
                exit(EXIT_FAILURE);
        } else {
                printf("\nCreated I/O context.\n");
        }
	
	datasize=1024*128*atoi(argv[1]);
   	data = (double*) malloc (datasize*sizeof(double));
    data1 = (double*) malloc (datasize*sizeof(double));
   	if (data==NULL || data1 ==NULL){
        	fprintf(stderr,"malloc failed.\n");
        	return -1;
   	}
    time_t cur;
    cur = time(0);
    printf("cur=%d\n",cur);
    srand(1000);
   	for (int i =0;i<datasize;i++){
            //printf("%d\n",i);
            if (i < datasize){
        		data[i]=(double)rand()/(double)(RAND_MAX/10);
                //printf("data[%d]=%f\n",i,data[11]);
                data1[i]=(double)(10.1233423112); 
                //printf("(double)rand()=%f\n",(double)rand()/(double)(RAND_MAX/10));
            }
            else
   	        	data[i]=(double)(10);
            //printf("data[%d]=%f\n",i,data[i]);
    }
    //printf("data[%d]=%f\n", datasize/2+datasize/4, data[datasize/2+datasize/4]); 
	//err = rados_write(io, "test_sample", data, datasize*sizeof(double), 0);
    printf("wrting\n");
    char object_name;
    for (int i=0; i < 20; i++){
        datasize = 1024*1024*i; 
        sprintf(object_name, "%s", datasize*8);
    	
    	err = rados_write(io, object_name, data, datasize*sizeof(double), 0);
    	//err = rados_write(io, "all_same", data, datasize*sizeof(double), 0);
    	//printf("Finish writing\n");
    //err = rados_write(io, "all_same", data1, datasize*sizeof(double), 0);
    }
	rados_ioctx_destroy(io);
	rados_shutdown(cluster);
	
	return 0;
}		
