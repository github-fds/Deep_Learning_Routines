SHELL=/bin/bash

DIR_OBJ   := ./obj
DUMMY     := $(shell [ -d $(DIR_OBJ) ] || mkdir $(DIR_OBJ) )

DIR_LIB   := ../lib
DUMMY     := $(shell [ -d $(DIR_LIB) ] || mkdir $(DIR_LIB) )

DIR_INC   := ../include
DUMMY     := $(shell [ -d $(DIR_INC) ] || mkdir $(DIR_INC) )

LIB       := dlr

DIR_SRC   := .
C_SRCS    := $(DIR_SRC)/dlr_common.c
CPP_SRCS  :=\
             $(DIR_SRC)/activation_leakyrelu.cpp\
             $(DIR_SRC)/activation_relu.cpp\
             $(DIR_SRC)/activation_sigmoid.cpp\
             $(DIR_SRC)/activation_tanh.cpp\
             $(DIR_SRC)/concat_2d.cpp\
             $(DIR_SRC)/convolution_2d.cpp\
             $(DIR_SRC)/deconvolution_2d.cpp\
             $(DIR_SRC)/linear_1d.cpp\
             $(DIR_SRC)/linear_nd.cpp\
             $(DIR_SRC)/norm_1d_batch.cpp\
             $(DIR_SRC)/norm_2d_batch.cpp\
             $(DIR_SRC)/norm_3d_batch.cpp\
             $(DIR_SRC)/pooling_2d_avg.cpp\
             $(DIR_SRC)/pooling_2d_max.cpp
C_HDRS    :=\
             $(DIR_SRC)/activation_leakyrelu.h\
             $(DIR_SRC)/activation_relu.h\
             $(DIR_SRC)/activation_sigmoid.h\
             $(DIR_SRC)/activation_tanh.h\
             $(DIR_SRC)/concat_2d.h\
             $(DIR_SRC)/convolution_2d.h\
             $(DIR_SRC)/deconvolution_2d.h\
             $(DIR_SRC)/linear_1d.h\
             $(DIR_SRC)/linear_nd.h\
             $(DIR_SRC)/norm_1d_batch.h\
             $(DIR_SRC)/norm_2d_batch.h\
             $(DIR_SRC)/norm_3d_batch.h\
             $(DIR_SRC)/pooling_2d_avg.h\
             $(DIR_SRC)/pooling_2d_max.h
CPP_HDRS  :=\
             $(DIR_SRC)/activation_leakyrelu.hpp\
             $(DIR_SRC)/activation_relu.hpp\
             $(DIR_SRC)/activation_sigmoid.hpp\
             $(DIR_SRC)/activation_tanh.hpp\
             $(DIR_SRC)/concat_2d.hpp\
             $(DIR_SRC)/convolution_2d.hpp\
             $(DIR_SRC)/deconvolution_2d.hpp\
             $(DIR_SRC)/linear_1d.hpp\
             $(DIR_SRC)/linear_nd.hpp\
             $(DIR_SRC)/norm_1d_batch.hpp\
             $(DIR_SRC)/norm_2d_batch.hpp\
             $(DIR_SRC)/norm_3d_batch.hpp\
             $(DIR_SRC)/pooling_2d_avg.hpp\
             $(DIR_SRC)/pooling_2d_max.hpp
OBJS      := $(addprefix $(DIR_OBJ)/,$(patsubst %.c,%.o,$(notdir $(C_SRCS))))
OBJS      += $(addprefix $(DIR_OBJ)/,$(patsubst %.cpp,%.o,$(notdir $(CPP_SRCS))))

DEFINES   :=
CFLAGS    := -O3 -x c -c -fPIC -I$(DIR_SRC) $(DEFINES)
CPPFLAGS  := -O3 -c -fPIC -I$(DIR_SRC) $(DEFINES)

vpath %.h    $(DIR_SRC)
vpath %.c    $(DIR_SRC)
vpath %.hpp  $(DIR_SRC)
vpath %.cpp  $(DIR_SRC)

$(DIR_OBJ)/%.o: %.c %.h
	g++ $(CFLAGS) -o $@ $<

$(DIR_OBJ)/%.o: %.cpp %.hpp
	g++ $(CPPFLAGS) -o $@ $<

all: $(OBJS) lib$(LIB).so lib$(LIB).a

install: lib$(LIB).so lib$(LIB).a dlr.h dlr.hpp
	@if [ -f lib$(LIB).so ]; then\
		if [ ! -d $(DIR_LIB) ]; then mkdir $(DIR_LIB); fi;\
		install lib$(LIB).so lib$(LIB).a $(DIR_LIB);\
		if [ ! -d $(DIR_INC) ]; then mkdir $(DIR_INC); fi;\
		install $(DIR_SRC)/dlr_common.h $(C_HDRS) $(CPP_HDRS) $(DIR_INC);\
		install dlr.h dlr.hpp $(DIR_INC);\
	else\
		echo "lib$(LIB).so not found";\
	fi

lib$(LIB).so lib$(LIB).a: $(OBJS)
	g++ -shared -Wl,-soname,lib$(LIB) -o lib$(LIB).so $^
	ar rcs lib$(LIB).a $^

.PHONY: dlr.h dlr.hpp
dlr.h:
	@echo "#pragma once"              >  $@
	@echo "#if !defined(__SYNTHESIS__)" >> $@
	@echo "#include \"dlr_common.h\"" >> $@
	@echo "#endif" >> $@
	@for H in $(notdir $(C_HDRS)); do\
		echo "#include \"$$H\"" >> $@;\
	done

dlr.hpp:
	@echo "#pragma once"              >  $@
	@echo "#if !defined(__SYNTHESIS__)" >> $@
	@echo "#include \"dlr_common.h\"" >> $@
	@echo "#endif" >> $@
	@for H in $(notdir $(CPP_HDRS)); do\
		echo "#include \"$$H\"" >> $@;\
	done

.PHONY: all clean cleanup clobber cleanupall disclean

clean:
	/bin/rm   -f  *.o
	/bin/rm   -rf $(DIR_OBJ)
	/bin/rm   -f lib$(LIB).so lib$(LIB).a
	/bin/rm   -f dlr.h dlr.hpp

cleanup clobber: clean
	/bin/rm   -rf $(DIR_LIB)
	/bin/rm   -rf $(DIR_INC)

cleanupall distclean: cleanup
