SHELL=/bin/bash

ifndef DLR_HOME
    export DLR_HOME := ..
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
	fi
	make conv.2d
	make pool.2d.max
	make pool.2d.avg
	make linear.1d
	make linear.nd
	make concat.2d
	make activation
	make norm.2d.batch
	make norm.nd.batch

conv.2d: $(DIR_LIB)/$(LIB_SO)
	python3 modules/convolution_2d_wrapper.py

pool.2d.max: $(DIR_LIB)/$(LIB_SO)
	python3 modules/pooling_2d_max_wrapper.py

pool.2d.avg: $(DIR_LIB)/$(LIB_SO)
	python3 modules/pooling_2d_avg_wrapper.py

linear.1d: $(DIR_LIB)/$(LIB_SO)
	python3 modules/linear_1d_wrapper.py

linear.nd: $(DIR_LIB)/$(LIB_SO)
	python3 modules/linear_nd_wrapper.py

concat.2d: $(DIR_LIB)/$(LIB_SO)
	python3 modules/concat_2d_wrapper.py

activation: $(DIR_LIB)/$(LIB_SO)
	python3 modules/activation_wrapper.py

norm.2d.batch: $(DIR_LIB)/$(LIB_SO)
	python3 modules/norm_2d_batch_wrapper.py

norm.nd.batch: $(DIR_LIB)/$(LIB_SO)
	python3 modules/norm_3d_batch_wrapper.py

DIRS	= $(subst /,, $(dir $(wildcard */Makefile)))

clean:
	/bin/rm -f  *.o
	/bin/rm -f  *.so
	/bin/rm -f  *.pyc *.pyo
	/bin/rm -rf __pycache__
	for D in $(DIRS); do\
		if [ -f $$D/Makefile ] ; then \
			echo "make -C $$D -s $@";\
			make -C $$D -s $@;\
		fi;\
	done

cleanup clobber: clean
	for D in $(DIRS); do\
		if [ -f $$D/Makefile ] ; then \
			echo "make -C $$D -s $@";\
			make -C $$D -s $@;\
		fi;\
	done

cleanupall distclean: cleanup
	for D in $(DIRS); do\
		if [ -f $$D/Makefile ] ; then \
			echo "make -C $$D -s $@";\
			make -C $$D -s $@;\
		fi;\
	done

.PHONY: all clean cleanup clobber cleanupall
