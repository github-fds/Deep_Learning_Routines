SHELL=/bin/bash

ifndef DLR_HOME
    export DLR_HOME := ../..
endif

ifdef PYTHONPATH
  export PYTHONPATH=$(DLR_HOME):$(shell printenv PYTHONPATH)
else
  export PYTHONPATH=$(DLR_HOME)
endif

DIR_LIB := $(DLR_HOME)/lib
LIB_SO  := libdlr.so

all:

all_test run run_test:
	@if [ ! -f $(DIR_LIB)/$(LIB_SO) ]; then\
		echo "$(LIB_SO) not found";\
		exit 1;\
	fi
	@if [ ! -d $(DLR_HOME)/python ]; then\
		echo "$(DLR_HOME)/python not found";\
		exit 1;\
	fi
	make conv.2d
	make pool.2d.max
	make pool.2d.avg
	make linear.1d
	make linear.nd
	make concat.2d
	make activations
#	make norm.batch
#	make deconv.2d

help: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --help

conv.2d: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --dtype=float32 --limit=1E-1 --layer Convolution2d --rigor

pool.2d.max: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --dtype=float32 --limit=1E-2 --layer Pooling2dMax --rigor #--verbose

pool.2d.avg: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --dtype=float32 --limit=1E-3 --layer Pooling2dAvg --rigor

linear.1d: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py

linear.nd: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py

concat.2d: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py

activation.relu relu: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --layer ReLu --random --rigor

activation.leakyrelu leakyrelu: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --layer LeakyReLu --dtype=float32 --rigor --verbose

activation.all activations: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --dtype=float32 --layer ReLu      --rigor
	python dlr_pytorch_wrapper.py --dtype=float32 --layer LeakyReLu --rigor
	python dlr_pytorch_wrapper.py --dtype=float32 --layer Tanh      --rigor
	python dlr_pytorch_wrapper.py --dtype=float32 --layer Sigmoid   --rigor

norm.batch: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --dtype=float32 --layer NormBatch --rigor

deconv.2d: $(DIR_LIB)/$(LIB_SO)
	python dlr_pytorch_wrapper.py --dtype=float32 --limit=1E-1 --layer Deconvolution2d --rigor

DIRS	= $(subst /,, $(dir $(wildcard */Makefile)))

clean:
	/bin/rm -f  *.o
	/bin/rm -f  *.so
	/bin/rm -f  *.pyc *.pyo
	/bin/rm -rf __pycache__

cleanup clobber: clean

cleanupall distclean: cleanup

.PHONY: all clean cleanup clobber cleanupall
