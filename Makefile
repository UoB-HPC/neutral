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
OPTIONS         += -DTILES -DENABLE_PROFILING 

ifeq ($(DEBUG), yes)
  OPTIONS += -O0 -g -DDEBUG 
endif

ifeq ($(MPI), yes)
	CC = mpiicc
	OPTIONS += -DMPI
else
	CC = icc
endif

# Get specialised kernels
SRC  			 = $(wildcard *.c)
SRC  			+= $(wildcard $(KERNELS)/*.c)
SRC  			+= $(wildcard $(MULTI_DIR)/$(KERNELS)/*.c)
SRC 			+= $(subst main.c,, $(wildcard $(MULTI_DIR)/*.c))
SRC_CLEAN  = $(subst $(MULTI_DIR)/,,$(SRC))
OBJS 			+= $(patsubst %.c, $(MULTI_BUILD_DIR)/%.o, $(SRC_CLEAN))

bright: make_build_dir $(OBJS) Makefile
	$(MULTI_LINKER) $(OBJS) $(OPTIONS) $(MULTI_LDFLAGS) -o bright.$(KERNELS)

# Rule to make controlling code
$(MULTI_BUILD_DIR)/%.o: %.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

$(MULTI_BUILD_DIR)/%.o: $(MULTI_DIR)/%.c Makefile 
	$(MULTI_COMPILER_CC) $(MULTI_FLAGS) $(OPTIONS) -c $< -o $@

make_build_dir:
	@mkdir -p $(MULTI_BUILD_DIR)/
	@mkdir -p $(MULTI_BUILD_DIR)/$(KERNELS)

clean:
	rm -rf $(EXE).exe *.bov *.dat

