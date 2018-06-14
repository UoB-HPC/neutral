# User defined parameters
KERNELS  					 = omp4
COMPILER 					 = CRAY
MPI      					 = no
OPTIONS  					+= -DTILES #-DENABLE_PROFILING 
ARCH_COMPILER_CC   = cc
ARCH_COMPILER_CPP  = CC

# Compiler-specific flags
CFLAGS_INTEL			 = -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
										 -Wall -qopt-report=5 -xhost
CFLAGS_INTEL_KNL	 = -O3 -qopenmp -no-prec-div -std=gnu99 -DINTEL \
										 -xMIC-AVX512 -Wall -qopt-report=5
CFLAGS_GCC				 = -O3 -std=gnu99 -fopenmp -march=native -Wall
CFLAGS_GCCTX2			 = -O3 -std=gnu99 -fopenmp -Wall
CFLAGS_GCC_KNL   	 = -O3 -fopenmp -std=gnu99 \
										 -mavx512f -mavx512cd -mavx512er -mavx512pf
CFLAGS_GCC_POWER   = -O3 -mcpu=power8 -mtune=power8 -fopenmp -std=gnu99
CFLAGS_CRAY				 = -lrt -hlist=a
CFLAGS_XL					 = -O3 -qsmp=omp
CFLAGS_XL_OMP4		 = -qsmp -qoffload
CFLAGS_CLANG			 = -std=gnu99 -fopenmp=libiomp5 -march=native -Wall
CFLAGS_CLANG_OMP4  = -O3 -Wall -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-nonaliased-maps \
										 -fopenmp=libomp --cuda-path=$(CUDA_PATH) -DCLANG
CFLAGS_PGI_NV			 = -fast -acc -ta=tesla:cc60 -Minfo=acc
CFLAGS_PGI_MC			 = -ta=multicore -fast 

OPTIONS  					+= -D__STDC_CONSTANT_MACROS

ifeq ($(KERNELS), cuda)
  CHECK_CUDA_ROOT = yes
endif
ifeq ($(COMPILER), CLANG_OMP4)
  CHECK_CUDA_ROOT = yes
endif

ifeq ($(CHECK_CUDA_ROOT), yes)
ifeq ("${CUDA_PATH}", "")
$(error "$$CUDA_PATH is not set, please set this to the root of your CUDA install.")
endif
endif

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -DDEBUG -g
endif

ifeq ($(MPI), yes)
  OPTIONS += -DMPI
endif

# Default compiler
ARCH_LINKER    		= $(ARCH_COMPILER_CC)
ARCH_FLAGS     		= $(CFLAGS_$(COMPILER))
ARCH_LDFLAGS   		= $(ARCH_FLAGS) -lm
ARCH_BUILD_DIR 		= ../obj/neutral/
ARCH_DIR       		= ..

ifeq ($(KERNELS), cuda)
  include Makefile.cuda
  OPTIONS += -DSoA
endif

ifeq ($(KERNELS), omp4)
  OPTIONS += -DSoA
endif

ifeq ($(KERNELS), oacc)
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
		*.dat *.optrpt *.cub *.ptx *.ap2 *.xf *.ptx1

