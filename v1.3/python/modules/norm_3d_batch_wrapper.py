#!/usr/bin/env python
"""
This file contains Python interface of norm_3d_batch.
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
__description__= "Python interface of norm_3d_batch"

#-------------------------------------------------------------------------------
import ctypes
import ctypes.util
import numpy as np
import python.modules.dlr_common as dlr_common

#===============================================================================
def Norm3dBatch( out_data     # in_channel x <N dimemsion>
               , in_data      # in_channel x <N dimemsion>
               , running_mean # in_channel
               , running_var  # in_channel
               , scale=None   # None or in_channel
               , bias=None    # None or in_channel    
               , epsilon=1E-5
               , rigor=False
               , verbose=False):
    """
    Returns True on success, otherwize returns False
    Applies a 1D matrix multiplication over an input data data.
    Note that all nd-array lists are NumPy (mutable), not PyTorch tensor (immutable).
    :param out_data: <mutable> output data, out_data[ndim][out_size]
    :param in_data: input data, in_data[ndim][in_size]
    :param weight: weight[out_size][in_size]
    :param bias: bias for each output, bias[out_size]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: 'True' on success, 'False' on failure.
    Follwoings are derived from input arguments
    . ndim: first dimension of out/in_data
    . out_size: array size of out_data
    . in_size: array size of in_data
    . weight_size: dimension of weight
    . bias_size: array size of bias
    Following is an example usage for PyTorch.
        Norm2dBatch( tensor_out_data.data.numpy() # ndim x out_size
                       , tenso_in_data.data.numpy()   # ndim x in_size
                       , tensor_running_mean.data.numpy()   # out_size x in_size
                       , tensor_running_var.data.numpy()     # out_size
                       , tensor_scale.data.numpy()     # out_size
                       , tensor_bias.data.numpy()     # out_size
                       , epsilon
                       , rigor=True
                       , verbose=True)
    """
    if rigor:
       error =0
       if (out_data.ndim!=in_data.ndim):
           error += 1
           if verbose: dlr_common.DpuError("out_data in_data dimension mis-match", flush=True)
       if (running_mean.ndim!=1):
           error += 1
           if verbose: dlr_common.DpuError("running_mean dimension mis-match", flush=True)
       if (running_mean.size!=in_data.shape[0]):
           error += 1
           if verbose: dlr_common.DpuError("running_mean size mis-match", flush=True)
       if (running_var.ndim!=1):
           error += 1
           if verbose: dlr_common.DpuError("running_var dimension mis-match", flush=True)
       if (running_var.size!=in_data.shape[0]):
           error += 1
           if verbose: dlr_common.DpuError("running_var size mis-match", flush=True)
       if (scale is not None) and (scale.ndim!=1):
           error += 1
           if verbose: dlr_common.DpuError(f"scale should be 1 dim: {scale.ndim}", flush=True)
       if (bias is not None) and (bias.ndim!=1):
           error += 1
           if verbose: dlr_common.DpuError(f"bias should be 1 dim: {bias.ndim}", flush=True)
       t_out_channel     = out_data.shape[0]
       t_out_size        = out_data.size/t_out_channel
       t_in_channel      = in_data.shape[0]
       t_in_size         = in_data.size/t_in_channel
       if (t_out_channel!=t_in_channel):
           error += 1
           dlr_common.DpuError(f"channel mis-match", flush=True)
       if (t_out_size!=t_in_size):
           error += 1
           dlr_common.DpuError(f"channel mis-match", flush=True)
       if verbose:
          dlr_common.DpuInfo(f"out_data   ={out_data.shape}")
          dlr_common.DpuInfo(f"in_data    ={in_data.shape}")
       if (error!=0):
           dlr_common.DpuError("parameter mis-match", flush=True)
           return False
    #_fname=''
    #_ctype=''
    if out_data.dtype.type == np.int32:
        _fname = 'Norm3dBatchInt'
        _ctype = ctypes.c_int
    elif out_data.dtype.type == np.float32:
        _fname = 'Norm3dBatchFloat'
        _ctype = ctypes.c_float
    elif out_data.dtype.type == np.float64:
        _fname = 'Norm3dBatchDouble'
        _ctype = ctypes.c_double
    else:
        dlr_common.DpuError(" not support "+str(out_data.dtype.type), flush=True)
        return False
    _Norm3dBatch=dlr_common.WrapFunction(dlr_common._dlr
                                 ,_fname
                                 , None          # return type
                                 ,[ctypes.POINTER(_ctype) # out data
                                  ,ctypes.POINTER(_ctype) # in data
                                  ,ctypes.POINTER(_ctype) # running_mean
                                  ,ctypes.POINTER(_ctype) # running_var
                                  ,ctypes.POINTER(_ctype) # scale
                                  ,ctypes.POINTER(_ctype) # bias
                                  ,ctypes.c_uint    # in_size
                                  ,ctypes.c_ushort  # scale_size
                                  ,ctypes.c_ushort  # bias_size
                                  ,ctypes.c_ushort  # in_channel
                                  ,ctypes.c_float   # epsilon
                                  ,ctypes.c_int     # rigor
                                  ,ctypes.c_int ])  # verbose
    in_channel = in_data.shape[0]
    in_size    = int(in_data.size/in_channel) # num of elements per channel
    CP_out_data    = out_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_data     = in_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_running_mean= running_mean.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_running_var = running_var.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_size     = ctypes.c_uint(in_size)
    CP_in_channel  = ctypes.c_ushort(in_channel)
    CP_epsilon     = ctypes.c_float(epsilon)
    CP_rigor       = 1 if rigor else 0
    CP_verbose     = 1 if verbose else 0
    if (scale is None) or (scale.size == 0):
       CP_scale       = ctypes.POINTER(_ctype)()
       CP_scale_size  = ctypes.c_ushort(0)
    else:
       CP_scale       = scale.ctypes.data_as(ctypes.POINTER(_ctype))
       CP_scale_size  = ctypes.c_ushort(scale.shape[0])
    if (bias is None) or (bias.size == 0):
       CP_bias        = ctypes.POINTER(_ctype)()
       CP_bias_size   = ctypes.c_ushort(0)
    else:
       CP_bias        = bias.ctypes.data_as(ctypes.POINTER(_ctype))
       CP_bias_size   = ctypes.c_ushort(bias.shape[0])
    _Norm3dBatch(CP_out_data
                ,CP_in_data
                ,CP_running_mean
                ,CP_running_var
                ,CP_scale
                ,CP_bias
                ,CP_in_size
                ,CP_scale_size
                ,CP_bias_size
                ,CP_in_channel
                ,CP_epsilon
                ,CP_rigor
                ,CP_verbose)
    return True

#===============================================================================
if __name__=='__main__':
    def TestNorm3dBatch(_dtype):
        """
        _dtype: specify data type of data one of {np.int32, np.float32, np.float64}
        """
        in_size    = [ 1, 2, 3 ] # [ in_channel, ....]
        in_data = (100+100)*np.random.random(size=in_size)-100
        out_data = np.empty(in_size, dtype=_dtype)
        running_mean = (100+100)*np.random.random(size=(in_size[0]))-100
        running_var  = (100+100)*np.random.random(size=(in_size[0]))-100
        scale        = (10+10)*np.random.random(size=(in_size[0]))-10
        bias         = (10+10)*np.random.random(size=(in_size[0]))-10
        if _dtype is np.int32:
            in_data = np.int32(in_data)
            running_mean = np.int32(running_mean)
            running_var  = np.int32(running_var)
            scale        = np.int32(scale)
            bias         = np.int32(bias)
        epsilon = 1e-5
        status = Norm3dBatch( out_data
                            , in_data
                            , running_mean=running_mean
                            , running_var =running_var
                            , scale       =scale
                            , bias        =bias
                            , epsilon     =epsilon
                            , rigor=True
                            , verbose=True
                            )
        if status:
            dlr_common.DpuPrint(f"out_data:\n{out_data}", flush=True)
            dlr_common.DpuPrint(f"in_data:\n{in_data}", flush=True)
            dlr_common.DpuPrint(f"running_mean:\n{running_mean}", flush=True)
            dlr_common.DpuPrint(f"running_var:\n{running_var}", flush=True)
            dlr_common.DpuPrint(f"scale:\n{scale}", flush=True)
            dlr_common.DpuPrint(f"bias:\n{bias}", flush=True)
            dlr_common.DpuPrint(f"epsilon:\n{epsilon}", flush=True)

#===============================================================================
if __name__=='__main__':
    dlr_common.DpuPrint("Testing Norm3dBatch", flush=True)
    dlr_common.DpuPrint("*********************", flush=True)
    #TestNorm3dBatch(_dtype=np.int32)
    TestNorm3dBatch(_dtype=np.float32)
    #TestNorm3dBatch(_dtype=np.float64)

#===============================================================================
# Revision history:
#
# 2020.09.30: argument order of bias and bias_size changed
# 2020.04.25: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
