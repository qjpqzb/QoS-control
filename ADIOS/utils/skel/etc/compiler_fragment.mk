CC = mpicc
CFLAGS = -g -O3 -fPIC -fopenmp
FC = mpif90
FCFLAGS = -g -O2
FCLIBS =  -L/usr/lib -L/usr/lib/gcc/x86_64-linux-gnu/5 -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../../lib -L/lib/x86_64-linux-gnu -L/lib/../lib -L/usr/lib/x86_64-linux-gnu -L/usr/lib/../lib -L/usr/lib/gcc/x86_64-linux-gnu/5/../../.. -lrados -lgfortran -lm -lquadmath -lpthread
LDFLAGS = -L/usr/lib -pthread
prefix = /usr/local
