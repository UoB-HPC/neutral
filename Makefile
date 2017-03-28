# User defined parameters
KERNELS          = omp3
COMPILER         = INTEL_KNL
MPI              = yes
MAC_RPATH				 = -Wl,-rpath,${COMPILER_ROOT}/lib 
CFLAGS_INTEL     = -O3 -no-prec-div -std=gnu99 -qopenmp -DINTEL \
									 $(MAC_RPATH) -Wall -qopt-report=5 -g #-xhost
CFLAGS_INTEL_KNL = -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
									 -xMIC-AVX512 -Wall -restrict -g -DSoA #-qopt-report=5 
CFLAGS_GCC       = -O3 -g -std=gnu99 -fopenmp -march=native -Wall #-std=gnu99
CFLAGS_CRAY      = -lrt -hlist=a
OPTIONS         += -DTILES -D__STDC_CONSTANT_MACROS #-DENABLE_PROFILING #-DVISIT_DUMP

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -DDEBUG 
endif

ifeq ($(MPI), yes)
  OPTIONS += -DMPI
endif

# Default compiler
MULTI_COMPILER_CC   = mpiicc
MULTI_COMPILER_CPP  = mpiicpc
MULTI_LINKER    		= $(MULTI_COMPILER_CC)
MULTI_FLAGS     		= $(CFLAGS_$(COMPILER))
MULTI_LDFLAGS   		= $(MULTI_FLAGS) #-lm
MULTI_BUILD_DIR 		= ../obj
MULTI_DIR       		= ..

ifeq ($(KERNELS), cuda)
include Makefile.cuda
OPTIONS += -DSoA
endif

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(MULTI_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(MULTI_DIR)/*.c))
SRC_CLEAN  = $(subst $(MULTI_DIR)/,,$(SRC))
OBJS 			+= $(patsubst %.c, $(MULTI_BUILD_DIR)/%.o, $(SRC_CLEAN))

neutral: make_build_dir $(OBJS) Makefile
	$(MULTI_LINKER) $(OBJS) $(OPTIONS) $(MULTI_LDFLAGS) -o neutral.$(KERNELS)

# Rule to make controlling code
$(MULTI_BUILD_DIR)/%.o: %.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

$(MULTI_BUILD_DIR)/%.o: $(MULTI_DIR)/%.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(MULTI_BUILD_DIR)/
	@mkdir -p $(MULTI_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(MULTI_BUILD_DIR)/* neutral.$(KERNELS) *.vtk *.bov *.dat *.optrpt *.cub *.ptx *.ap2 *.xf

