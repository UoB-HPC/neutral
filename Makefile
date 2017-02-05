# User defined parameters
KERNELS          = omp3
COMPILER         = INTEL
MPI              = yes
MAC_RPATH				 = -Wl,-rpath,${COMPILER_ROOT}/lib 
CFLAGS_INTEL     = -O3 -no-prec-div -std=gnu99 -qopenmp -DINTEL \
									 $(MAC_RPATH) -xhost -Wall -qopt-report=5
CFLAGS_INTEL_KNL = -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
									 -xMIC-AVX512 -Wall -qopt-report=5
CFLAGS_GCC       = -O3 -g -std=gnu99 -fopenmp -march=native -Wall #-std=gnu99
CFLAGS_CRAY      = -lrt -hlist=a
OPTIONS          = -DTILES -DENABLE_PROFILING 

MPI     = no
OPTIONS = -g -DENABLE_PROFILING -qopt-report=5
CFLAGS  = -O3 -std=gnu99 -xhost #-qopenmp
LDFLAGS = #-lrt
EXE			= bright

ifeq ($(MPI), yes)
	CC = mpiicc
	OPTIONS += -DMPI
else
	CC = icc
endif

all:
	$(CC) $(CFLAGS) $(OPTIONS) $(LDFLAGS) main.c profiler.c -o $(EXE).exe

clean:
	rm -rf $(EXE).exe *.bov *.dat

