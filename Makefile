SOURCE = $(wildcard *.c)
TARGETS = $(patsubst %.c, %, $(SOURCE))
 
CC = gcc
CFLAGS = -lrados
 
all:$(TARGETS)
 
$(TARGETS):%:%.c
	$(CC) $< $(CFLAGS) -o $@
 
.PHONY:clean all
clean:
	-rm -rf $(TARGETS)










#interference: interference.c
#	gcc interference.c -lrados -o interference
#interference1: interference1.c
	#gcc interference1.c -lrados -o interference1
#write_sample: write_sample.c
	#gcc write_sample.c -lrados -o write_sample  
