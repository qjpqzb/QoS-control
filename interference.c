#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rados/librados.h>
#include <time.h>
#include <sys/time.h>
int main (int argc, const char **argv)
{
        struct timeval ts,tf;
        double results[512]={0};
        gettimeofday(&ts,NULL);
        /* Declare the cluster handle and required arguments. */
        rados_t cluster;
        char cluster_name[] = "ceph";
        char user_name[] = "client.3";
        uint64_t flags = 0;
	double * data=0;
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
        char buf[32];
        rados_conf_get(cluster,"bluestore default buffered read",buf,32);
        printf("bluestore default buffered read=%s\n",buf);
        //rados_conf_get(cluster,"osd max write size",buf,32);
        //printf("osd max write size=%s\n",buf);
	    rados_ioctx_t io;
        char *poolname = "noise2";

        err = rados_ioctx_create(cluster, poolname, &io);
        if (err < 0) {
                fprintf(stderr, "%s: cannot open rados pool %s: %s\n", argv[0], poolname, strerror(-err));
                rados_shutdown(cluster);
                exit(EXIT_FAILURE);
        } else {
                printf("\nCreated I/O context.\n");
        }
	
	datasize=1024*128*atof(argv[1]);
   	data = (double*) malloc (datasize*sizeof(double));
   	if (data==NULL){
        	fprintf(stderr,"malloc failed.\n");
        	return -1;
   	}
   	for (int i =0;i<datasize;i++){
        	data[i]=1.1234333678899086554;
   	}
    struct timeval tb,ta;
    char prename[16]="interference";
    printf("Interference start!\n");
    gettimeofday(&tf,NULL);
    //results[0] = 0.0;
    //results[1] = ((tf.tv_sec*1000 + tf.tv_usec/1000)-(ts.tv_sec*1000 + ts.tv_usec/1000))/1000.0;
    //results[2] = results[1]+0.0001;
    //printf("results[0]=%f\n",results[32]);
    //double sum=results[1];
    //int j=3;
    char objectname[16];
	double write_time = 0.0;
	for (int i=0; i<440; i++){
        sprintf(objectname, "%s%d", prename, i);
        printf("Object name=%s\n",objectname);
        gettimeofday(&tb,NULL);
        //printf("2\n");
        
        //for (int j=0; j<8; j++){
            //char temp[32]="";
            //char object_name[32] = "temp";
            //sprintf(temp,"%d",j);
            //strcat(object_name,temp);
        //printf("Before\n");
        //err = rados_write(io, object_name, data, datasize*sizeof(double),0);
        //printf("Before wrote\n");
        //printf("%d\n",i);
		err = rados_write(io, objectname, data, datasize*sizeof(double), 0);
        //printf("err=%d\n",err);
            //printf("%s\n",object_name); 
        //}
        //if (err < 0) {
        //    printf("123\n");
        //}
        //printf("After\n");
        //err = rados_write(io, "temp", data, sizeof(double)*datasize, 0);
        gettimeofday(&ta,NULL);
        //printf("sleep=%d\n",(int)((atof(argv[2])*coe)-(end-start)*coe/(double)CLOCKS_PER_SEC));
        //if ((end-start)*coe/(double)CLOCKS_PER_SEC > atof(argv[2])*coe)
        //    printf("The time of one write exceed the time you enter\n");
		//usleep((int)((atof(argv[2])*coe)-(end-start)*coe/(double)CLOCKS_PER_SEC));
        //printf("write s = %f\n",((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000))/1000.0);
		write_time = (ta.tv_sec + ta.tv_usec/1000.0/1000.0)-(tb.tv_sec + tb.tv_usec/1000.0/1000.0);
        printf("write s = %f\n",write_time);
        printf("Bandwidth = %f\n",atof(argv[1])/write_time);
        //printf("ms=%f\n",atof(argv[2])*1000-((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000)));       
        //sum+=((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000))/1000.0;
        //results[j] = sum;
        //results[j+1] = sum+0.0001;
        //sum+=10.0-((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000))/1000.0;
        //results[j+2] = sum;
        //results[j+3] = sum+0.0001;
        //j+=4;

		/*
        if (((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000))>(atof(argv[2])*1000)){
            printf("Time of write is larger than expected period!\n");
        }
        usleep(1000*(int)(atof(argv[2])*1000-((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000))));
		*/
    }
    FILE *fpWrite=fopen("noise1.txt","w"); 
    if(fpWrite==NULL)  
    {  
        return 0; 
    }
    for(int j=0;j<32;j++)  
        fprintf(fpWrite,"%f%s ",results[j],",");  
    fclose(fpWrite);      
    printf("Finish noise1\n");
	rados_ioctx_destroy(io);
	rados_shutdown(cluster);
	
	return 0;
}		
