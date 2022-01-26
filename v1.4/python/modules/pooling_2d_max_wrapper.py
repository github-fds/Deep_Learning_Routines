#!/usr/bin/env python
"""
This file contains Python interface of polling_2d_max.
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
__description__= "Python interface of pooling_2d_max"

#-------------------------------------------------------------------------------
import ctypes
import ctypes.util
import numpy as np
import math
from python.modules import dlr_common

#===============================================================================
def GetOutputSizeOfPooling2dMax( in_size
                               , kernel_size
                               , stride
                               , padding
                               , ceil_mode=False
                               , rigor=False
                               , verbose=False):
    """
    Returns satus and the size of output tensor
    :param in_size:
    :param kernel_size:
    :param stride:
    :param padding:
    :param ceil_mode: use floor when false, otherwize cel when true
    :param rigor:
    :param verbose:
    :return: the size of output tensor
    """
    err = 0
    if rigor:
       if (in_size<1):          
           err+=1
           if verbose: dlr_common.DlrError(f"in_size should be positive: {in_size}", flush=True)
       if (kernel_size<1):      
           err+=1
           if verbose: dlr_common.DlrError(f"kernel_size should be positive: {kernel_size}", flush=True)
       if ((kernel_size%2)==1): 
           err+=1
           if verbose: dlr_common.DlrError(f"kernel_size should be even: {kernel_size}", flush=True)
       if (stride<1):           
           err+=1
           if verbose: dlr_common.DlrError(f"stride should be larger than 0: {stride}", flush=True)
       if (padding<0):          
           err+=1
           if verbose: dlr_common.DlrError(f"padding should be positive: {padding}", flush=True)
    if ceil_mode: out_size = math.ceil(((in_size-kernel_size+2*padding)/stride)+1)
    else:         out_size = math.floor(((in_size-kernel_size+2*padding)/stride)+1)
    if err>0: return False, out_size
    else:     return True, out_size

#===============================================================================
def Pooling2dMax( out_data    # out_channel x out_size x out_size
                , in_data     # in_channel x in_size x in_size
                , kernel_size # kernel_size x kernel_size
                , stride=1
                , padding=0
                , ceil_mode=False
                , rigor=False
                , verbose=False):
    """
    Returns True on success, otherwize returns False
    Applies a 2D mAXpolling over an input data composed of several input channels.
    Note that all nd-array lists are NumPy (mutable), not PyTorch tensor (immutable).
    :param out_data: <mutable> output data, out_data[out_channel][out_size][out_size]
    :param in_data: input data, in_data[in_channel][in_size][in_size]
    :param kernel_size:
    :param stride: num of skips to apply next filter
    :param padding: num of pixes at the boundary
    :param ceil_mode: use floor() when false, otherwize ceil()
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: 'True' on success, 'False' on failure.
    Follwoings are derived from input arguments
    . out_size: array size of out_data
    . in_size: array size of in_data
    . in_chnannels: num of input channels
    . out_channels: num of output channels (it should be the same as in_channel)
    Following is an example usage for PyTorch.
        Pooling2dMax( tensor_out_data.data.numpy() # out_channel x out_size x out_size
                        , tenso_in_data.data.numpy()   # in_channel x in_size x in_size
                        , kernel_size
                        , stride
                        , padding
                        , rigor=True
                        , verbose=True)
    """
    if rigor:
       error =0
       if (out_data.ndim!=3):
           error += 1
           if verbose: dlr_common.DlrError("out_data is not 3 dim")
       if (in_data.ndim!=3): 
           error += 1
           if verbose: dlr_common.DlrError("in_data is not 3 dim")
       if (kernel_size<2): 
           error += 1
           if verbose: dlr_common.DlrError("kernel_size should be >=2")
       if (stride<1): 
           error += 1
           if verbose: dlr_common.DlrError("stride should be >=1")
       if (padding<0): 
           error += 1
           if verbose: dlr_common.DlrError("stride should be >=0")
       t_out_size    = out_data.shape[2] # note ndim (i.e., rank) is 3
       t_in_size     = in_data.shape[2] # note ndim (i.e., rank) is 3
       t_kernel_size = kernel_size;
       t_in_channel  = in_data.shape[0]
       t_out_channel = out_data.shape[0]
       t_stride      = stride
       t_padding     = padding
       if (t_in_channel!=t_out_channel):
           error += 1
           if verbose: dlr_common.DlrError("in/out channel should be the same")
       status, t_out_size_expect = GetOutputSizeOfPooling2dMax( t_in_size
                                                              , t_kernel_size
                                                              , t_stride
                                                              , t_padding )
       if not status: return False # something wrong with arguments
       if (t_out_size!=t_out_size_expect):
           error += 1
           if verbose: dlr_common.DlrError(f"out_size mis-match {t_out_size} {t_out_size_expect}")
       if ((t_kernel_size%2)==1):
           error += 1
           if verbose: dlr_common.DlrError(f"kernel_size should be even")
       if verbose:
          dlr_common.DlrInfo(f"out_channel={t_out_channel} {out_data.shape}")
          dlr_common.DlrInfo(f"in_channel ={t_in_channel} {in_data.shape}")
          dlr_common.DlrInfo(f"out_size   ={t_out_size} {out_data.shape}")
          dlr_common.DlrInfo(f"in_size    ={t_in_size} {in_data.shape}")
          dlr_common.DlrInfo(f"kernel_size={t_kernel_size}")
          dlr_common.DlrInfo(f"stride     ={t_stride} {stride}")
          dlr_common.DlrInfo(f"padding    ={t_padding} {padding}")
       if (error!=0):
           dlr_common.DlrError("parameter mis-match");
           return False
    #_fname=''
    #_ctype=''
    if out_data.dtype.type == np.int32:
        _fname = 'Pooling2dMaxInt'
        _ctype = ctypes.c_int
    elif out_data.dtype.type == np.float32:
        _fname = 'Pooling2dMaxFloat'
        _ctype = ctypes.c_float
    elif out_data.dtype.type == np.float64:
        _fname = 'Pooling2dMaxDouble'
        _ctype = ctypes.c_double
    else:
        dlr_common.DlrError("not support "+str(out_data.dtype.type))
        return False
    _Pooling2dMax=dlr_common.WrapFunction(dlr_common._dlr
                                  ,_fname
                                  , None          # return type
                                  ,[ctypes.POINTER(_ctype) # output features
                                   ,ctypes.POINTER(_ctype) # input image
                                   ,ctypes.c_ushort  # out_size
                                   ,ctypes.c_ushort  # in_size
                                   ,ctypes.c_ubyte   # kernel_size (only for square filter)
                                   ,ctypes.c_ushort  # channel
                                   ,ctypes.c_ubyte   # stride
                                   ,ctypes.c_ubyte   # padding
                                   ,ctypes.c_int     # ceil_mode
                                   ,ctypes.c_int     # rigor
                                   ,ctypes.c_int     # verbose
                                   ]) 
    CP_out_data    = out_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_data     = in_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_out_size    = ctypes.c_ushort(out_data.shape[2]) # note ndim (i.e., rank) is 3
    CP_in_size     = ctypes.c_ushort(in_data.shape[2]) # note ndim (i.e., rank) is 3
    CP_kernel_size = ctypes.c_ubyte(kernel_size)
    CP_channel     = ctypes.c_ushort(in_data.shape[0])
    CP_stride      = ctypes.c_ubyte(stride)
    CP_padding     = ctypes.c_ubyte(padding)
    CP_ceil_mode   = 1 if ceil_mode else 0
    CP_rigor       = 1 if rigor else 0
    CP_verbose     = 1 if verbose else 0

    _Pooling2dMax(CP_out_data    
                 ,CP_in_data      
                 ,CP_out_size    
                 ,CP_in_size     
                 ,CP_kernel_size 
                 ,CP_channel  
                 ,CP_stride      
                 ,CP_padding     
                 ,CP_ceil_mode
                 ,CP_rigor
                 ,CP_verbose
                 )
    return True

#===============================================================================
# # Testing function
# def _Convolution2dRef_not_yet( out_data    # out_channel x out_size x out_size
#                              , in_data     # in_channel x in_size x in_size
#                              , kernel      # in_channel x out_channel x kernel_size x kernel_size
#                              , stride=1
#                              , padding=0
#                              , bias=0
#                              , rigor=True):
#     m, n = kernel.shape; # m=n
#     if (m == n):
#         y, x = image.shape
#         y = y - m + 1
#         x = x - m + 1
#         out_data = np.zeros((y,x))
#         for i in range(y):
#             for j in range(x):
#                 out_data[i][j] = np.sum(image[i:i+m, j:j+n]*kernel) + bias
#                 #out_data[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias

if __name__=='__main__':
    def TestPooling2dMax(_dtype):
        """
        _dtype: specify data type of data one of {np.int32, np.float32, np.float64}
        """
        in_channel = 1
        in_size = 8
        out_channel = in_channel
        kernel_size = 2
        stride = 1
        padding = 0
        ceil_mode = False
        in_data = np.empty([in_channel,in_size,in_size], dtype=_dtype)
        status, out_size = GetOutputSizeOfPooling2dMax( in_size
                                                      , kernel_size
                                                      , stride
                                                      , padding
                                                      , rigor=True
                                                      , verbose=True)
        if not status: return
        # prepare arrays to pass to C function
        out_data = np.empty([out_channel,out_size,out_size], dtype=_dtype)
        v = 0
        for i in range(in_channel):
            for r in range(in_size):
                for c in range(in_size):
                    in_data[i][r][c] = v    
                    v += 1

        status = Pooling2dMax( out_data    # out_channel x out_size x out_size
                             , in_data     # in_channel x in_size x in_size
                             , kernel_size # in_channel x out_channel x kernel_size x kernel_size
                             , stride
                             , padding
                             , ceil_mode
                             , rigor=True
                             , verbose=True)
        if status:
            dlr_common.DlrPrint(f"in_data:\n{in_data}")
            dlr_common.DlrPrint(f"out_data:\n{out_data}")

if __name__=='__main__':
    dlr_common.DlrPrint("Testing Pooling2dMax", flush=True);
    dlr_common.DlrPrint("*********************", flush=True)
    TestPooling2dMax(_dtype=np.int32)
    #TestPooling2dMax(_dtype=np.float32)
    #TestPooling2dMax(_dtype=np.float64)

#===============================================================================
# Revision history:
#
# 2020.04.58: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
