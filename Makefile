# User defined parameters
KERNELS          	 = omp3
COMPILER         	 = CRAY
MPI              	 = yes
MAC_RPATH				 	 = -Wl,-rpath,${COMPILER_ROOT}/lib 
CFLAGS_INTEL			 = -qopenmp -no-prec-div -std=gnu99 -DINTEL \
										 -Wall -qopt-report=5 #-xhost
CFLAGS_INTEL_KNL	 = -O3 -g -qopenmp -no-prec-div -std=gnu99 -DINTEL \
										 -xMIC-AVX512 -Wall -qopt-report=5
CFLAGS_GCC				 = -std=gnu99 -fopenmp -march=native -Wall #-std=gnu99
CFLAGS_GCC_POWER   = -O3 -g -mcpu=power8 -mtune=power8 -fopenmp -std=gnu99
CFLAGS_CRAY				 = -lrt -hlist=a
CFLAGS_XL					 = -O3 -qsmp=omp
CFLAGS_XL_OMP4		 = -qsmp -qoffload
CFLAGS_CLANG_OMP4  = -O3 -Wall -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-nonaliased-maps \
										 -fopenmp=libomp --cuda-path=/home/projects/pwr8-rhel73-lsf/cuda/8.0.44/ 
OPTIONS            += -DTILES -D__STDC_CONSTANT_MACROS #-DENABLE_PROFILING 

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -DDEBUG -g
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

