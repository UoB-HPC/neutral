# User defined parameters
KERNELS          		= omp3
COMPILER         		= INTEL
MPI              		= yes
MAC_RPATH				 		= -Wl,-rpath,${COMPILER_ROOT}/lib 
CFLAGS_INTEL     		= -O3 -no-prec-div -std=gnu99 -qopenmp -DINTEL \
								 		  $(MAC_RPATH) -Wall -qopt-report=5 -g #-xhost
CFLAGS_INTEL_KNL 		= -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
								 		  -xMIC-AVX512 -Wall -restrict -g #-qopt-report=5 
CFLAGS_GCC       		= -O3 -g -std=gnu99 -fopenmp -march=native -Wall #-std=gnu99
CFLAGS_CRAY      		= -lrt -hlist=a
CFLAGS_CLANG_OMP4   = -O3 -Wall -fopenmp-targets=nvptx64-nvidia-cuda \
										 -fopenmp=libomp --cuda-path=/nfs/modules/cuda/8.0.44/
OPTIONS            += -DTILES -D__STDC_CONSTANT_MACROS -DVISIT_DUMP#-DENABLE_PROFILING 

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -DDEBUG 
endif

ifeq ($(MPI), yes)
  OPTIONS += -DMPI
endif

# Default compiler
ARCH_COMPILER_CC   = mpicc
ARCH_COMPILER_CPP  = mpic++
ARCH_LINKER    		= $(ARCH_COMPILER_CC)
ARCH_FLAGS     		= $(CFLAGS_$(COMPILER))
ARCH_LDFLAGS   		= $(ARCH_FLAGS) -lm
ARCH_BUILD_DIR 		= ../obj/neutral/
ARCH_DIR       		= ..

ifeq ($(KERNELS), cuda)
  include Makefile.cuda
  OPTIONS += -DSoA
endif

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(ARCH_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(ARCH_DIR)/*.c))
SRC_CLEAN  = $(subst $(ARCH_DIR)/,,$(SRC))
OBJS 			+= $(patsubst %.c, $(ARCH_BUILD_DIR)/%.o, $(SRC_CLEAN))

neutral: make_build_dir $(OBJS) Makefile
	$(ARCH_LINKER) $(OBJS) $(OPTIONS) $(ARCH_LDFLAGS) -o neutral.$(KERNELS)

# Rule to make controlling code
$(ARCH_BUILD_DIR)/%.o: %.c Makefile 
	$(ARCH_COMPILER_CC) $(ARCH_FLAGS) $(OPTIONS) -c $< -o $@

$(ARCH_BUILD_DIR)/%.o: $(ARCH_DIR)/%.c Makefile 
	$(ARCH_COMPILER_CC) $(ARCH_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(ARCH_BUILD_DIR)/
	@mkdir -p $(ARCH_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(ARCH_BUILD_DIR)/* neutral.$(KERNELS) *.vtk *.bov \
		*.dat *.optrpt *.cub *.ptx *.ap2 *.xf

