#!/usr/bin/env python
"""
This file contains Python interface of activation functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#-------------------------------------------------------------------------------
__author__     = "Ando Ki, Chaeeon Lim"
__copyright__  = "Copyright 2020, Future Design Systems"
__credits__    = ["none", "some"]
__license__    = "FUTURE DESIGN SYSTEMS SOFTWARE END-USER LICENSE AGREEMENT"
__version__    = "0"
__revision__   = "1"
__maintainer__ = "Ando Ki"
__email__      = "contact@future-ds.com"
__status__     = "Development"
__date__       = "2020.09.30"
__description__= "Python interface of activation functions"

#-------------------------------------------------------------------------------
import ctypes
import ctypes.util
import numpy as np
import math
from python.modules import dlr_common

#===============================================================================
def Activations( func_name 
               , out_data # any dimension
               , in_data  # any dimension
               , negative_slope=0.01 # for LeakyReLu
               , rigor=False
               , verbose=False):
    """
    Returns True on success, otherwize returns False
    Applies a non-linear activation function over an input data composed of several input channels.
    Note that all nd-array lists are NumPy (mutable), not PyTorch tensor (immutable).
    :param out_data: <mutable> output data, out_data[...]
    :param in_data: input data, in_data[...]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: 'True' on success, 'False' on failure.
    Follwoings are derived from input arguments
    . out_size: array size of out_data
    . in_size: array size of in_data
    Following is an example usage for PyTorch.
        Activation'FUNC_NAME'( tensor_out_data.data.numpy() # contiguous array
                      , tenso_in_data.data.numpy() # contiguous array
                      , rigor=True
                      , verbose=True)
    """
    if rigor:
       error =0
       if (out_data.ndim!=in_data.ndim):
           error += 1
           if verbose: dlr_common.DlrError("data dimension mis-match")
       if (out_data.size!=in_data.size):
           error += 1
           if verbose: dlr_common.DlrError(f"data size mis-match {in_data.size} {out_data.size}")
       for dim in range(in_data.ndim):
           if (in_data.shape[dim]!=out_data.shape[dim]):
               error += 1
               if verbose: dlr_common.DlrError("data dimension size mis-match")
       if (error!=0):
           dlr_common.DlrError("parameter mis-match");
           return False
    if (out_data.ndim==0) or (out_data.ndim==1):
        channel = 1
        size = out_data.size
    elif (out_data.ndim==2):
        channel = out_data.shape[0]
        size = out_data.shape[1]
    else:
        channel = out_data.shape[0]
        size = np.prod(out_data.shape[1:])

    #_fname=''
    #_ctype=''
    if out_data.dtype.type == np.int32:
        _fname = 'Activation'+func_name+'Int'
        _ctype = ctypes.c_int
    elif out_data.dtype.type == np.float32:
        _fname = 'Activation'+func_name+'Float'
        _ctype = ctypes.c_float
    elif out_data.dtype.type == np.float64:
        _fname = 'Activation'+func_name+'Double'
        _ctype = ctypes.c_double
    else:
        dlr_common.DlrError("not support "+str(out_data.dtype.type))
        return False

    if func_name == 'LeakyReLu':
        _Activation=dlr_common.WrapFunction(dlr_common._dlr
                                    ,_fname
                                    , None          # return type
                                    ,[ctypes.POINTER(_ctype) # output
                                     ,ctypes.POINTER(_ctype) # input
                                     ,ctypes.c_uint    # number of elements
                                     ,ctypes.c_ushort  # number of channels
                                     ,ctypes.c_uint    # negative slope
                                     ,ctypes.c_int     # rigor
                                     ,ctypes.c_int     # verbose
                                     ]) 
        CP_out_data       = out_data.ctypes.data_as(ctypes.POINTER(_ctype))
        CP_in_data        = in_data.ctypes.data_as(ctypes.POINTER(_ctype))
        CP_size           = ctypes.c_uint(size)
        CP_channel        = ctypes.c_ushort(channel)
        CP_negative_slope = ctypes.c_uint.from_buffer(ctypes.c_float(negative_slope)).value
        CP_rigor          = 1 if rigor else 0
        CP_verbose        = 1 if verbose else 0

        _Activation(CP_out_data    
                   ,CP_in_data      
                   ,CP_size    
                   ,CP_channel
                   ,CP_negative_slope
                   ,CP_rigor
                   ,CP_verbose
                   )
    else :
        _Activation=dlr_common.WrapFunction(dlr_common._dlr
                                    ,_fname
                                    , None          # return type
                                    ,[ctypes.POINTER(_ctype) # output
                                     ,ctypes.POINTER(_ctype) # input
                                     ,ctypes.c_uint    # number of elements
                                     ,ctypes.c_ushort  # number of channels
                                     ,ctypes.c_int     # rigor
                                     ,ctypes.c_int     # verbose
                                     ]) 
        CP_out_data    = out_data.ctypes.data_as(ctypes.POINTER(_ctype))
        CP_in_data     = in_data.ctypes.data_as(ctypes.POINTER(_ctype))
        CP_size        = ctypes.c_uint(size)
        CP_channel     = ctypes.c_ushort(channel)
        CP_rigor       = 1 if rigor else 0
        CP_verbose     = 1 if verbose else 0

        _Activation(CP_out_data    
                   ,CP_in_data      
                   ,CP_size    
                   ,CP_channel
                   ,CP_rigor
                   ,CP_verbose
                   )
    return True

#===============================================================================
def ActivationReLu( out_data # any dimension
                  , in_data  # any dimension
                  , rigor=False
                  , verbose=False):
    return Activations( 'ReLu'
                      , out_data=out_data
                      , in_data=in_data
                      , rigor=rigor
                      , verbose=verbose)
def ActivationLeakyReLu( out_data # any dimension
                       , in_data  # any dimension
                       , negative_slope=0.01
                       , rigor=False
                       , verbose=False):
    return Activations( 'LeakyReLu'
                      , out_data=out_data
                      , in_data=in_data
                      , negative_slope=negative_slope
                      , rigor=rigor
                      , verbose=verbose)
def ActivationTanh( out_data # any dimension
                  , in_data  # any dimension
                  , rigor=False
                  , verbose=False):
    return Activations( 'Tanh'
                      , out_data=out_data
                      , in_data=in_data
                      , rigor=rigor
                      , verbose=verbose)
def ActivationSigmoid( out_data # any dimension
                     , in_data  # any dimension
                     , rigor=False
                     , verbose=False):
    return Activations( 'Sigmoid'
                      , out_data=out_data
                      , in_data=in_data
                      , rigor=rigor
                      , verbose=verbose)

#===============================================================================
if __name__=='__main__':
    def TestActivations(_dtype):
        """
        _dtype: specify data type of data one of {np.int32, np.float32, np.float64}
        """
        funcs = ["ReLu", "LeakyReLu", "Tanh", "Sigmoid"]
        for func in funcs:
            func_name = "Activation" + func
            dims = [1, 2]
            for dim in dims:
                ndim = np.random.randint(10, size=(dim))
                ndim += 1 # prevent 0 in dimension
                in_data = (100+100)*np.random.random(size=ndim)-100
                if _dtype is np.int32:
                    in_data = np.int32(in_data)
                out_data = np.empty(ndim, dtype=_dtype)
               #status = locals()[func_name]( out_data
                status = globals()[func_name]( out_data
                                             , in_data
                                             , rigor=True
                                             , verbose=True)
                if status:
                   dlr_common.DlrPrint(f"in_data\n{in_data}")
                   dlr_common.DlrPrint(f"out_data\n{out_data}")

if __name__=='__main__':
    dlr_common.DlrPrint("Testing Activations", flush=True);
    dlr_common.DlrPrint("*********************", flush=True)
    #TestActivations(_dtype=np.int32)
    TestActivations(_dtype=np.float32)
    #TestActivations(_dtype=np.float64)

#===============================================================================
# Revision history:
#
# 2020.04.58: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
