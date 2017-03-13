# User defined parameters
KERNELS          = omp3
COMPILER         = GCC
MPI              = yes
MAC_RPATH				 = -Wl,-rpath,${COMPILER_ROOT}/lib 
CFLAGS_INTEL     = -O3 -no-prec-div -std=gnu99 -qopenmp -DINTEL \
									 $(MAC_RPATH) -Wall -qopt-report=5 #-xhost
CFLAGS_INTEL_KNL = -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
									 -Wall -qopt-report=5 -restrict -xMIC-AVX512 
CFLAGS_GCC       = -O3 -fopenmp -march=native -Wall -std=gnu99 
CFLAGS_CRAY      = -lrt -hlist=a 
CFLAGS_CLANG     = -O3 -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda \
									--cuda-path=/nfs/modules/cuda/8.0.44/ \
									#-ffp-contract=fast -fopenmp-nonaliased-maps

OPTIONS         += -DTILES -DENABLE_PROFILING -D__STDC_CONSTANT_MACROS

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -DDEBUG -g
endif

ifeq ($(MPI), yes)
  OPTIONS += -DMPI
endif

# Default compiler
MULTI_COMPILER_CC   = mpicc
MULTI_COMPILER_CPP  = mpic++
MULTI_LINKER    		= $(MULTI_COMPILER_CC)
MULTI_FLAGS     		= $(CFLAGS_$(COMPILER))
MULTI_LDFLAGS   		= -lm 
MULTI_BUILD_DIR 		= ../obj
MULTI_DIR       		= ..

ifeq ($(KERNELS), cuda)
include Makefile.cuda
endif

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(MULTI_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(MULTI_DIR)/*.c))
SRC_CLEAN  = $(subst $(MULTI_DIR)/,,$(SRC))
OBJS 			+= $(patsubst %.c, $(MULTI_BUILD_DIR)/%.o, $(SRC_CLEAN))

bright: make_build_dir $(OBJS) Makefile
	$(MULTI_LINKER) $(OBJS) $(MULTI_FLAGS) $(MULTI_LDFLAGS) $(OPTIONS) -o bright.$(KERNELS)

# Rule to make controlling code
$(MULTI_BUILD_DIR)/%.o: %.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

$(MULTI_BUILD_DIR)/%.o: $(MULTI_DIR)/%.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(MULTI_BUILD_DIR)/
	@mkdir -p $(MULTI_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(MULTI_BUILD_DIR)/* bright.$(KERNELS) *.vtk *.bov *.dat *.optrpt *.cub *.ptx *.ap2 *.xf

