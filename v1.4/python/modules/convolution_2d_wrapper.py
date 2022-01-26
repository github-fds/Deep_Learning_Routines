#!/usr/bin/env python
"""
This file contains Python interface of convolution_2d.
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
__description__= "Python interface of convolution_2d"

#-------------------------------------------------------------------------------
import ctypes
import ctypes.util
import numpy as np
from python.modules import dlr_common

#===============================================================================
def GetOutputSizeOfConvolution2d( in_size
                                , kernel_size
                                , stride
                                , padding
                                , rigor=False
                                , verbose=False):
    """
    Returns satus and the size of output tensor
    :param in_size:
    :param kernel_size:
    :param stride:
    :param padding:
    :param rigor:
    :param verbose:
    :return: the size of output tensor
    """
    err = 0
    if rigor or dlr_common.rigor:
       if (in_size<1):
           err+=1
           if verbose: dlr_common.DlrError(f"in_size should be positive: {in_size}", flush=True)
       if (kernel_size<1):
           err+=1
           if verbose: dlr_common.DlrError(f"kernel_size should be positive: {kernel_size}", flush=True)
       if ((kernel_size%2)!=1):
           err+=1
           if verbose: dlr_common.DlrError(f"kernel_size should be odd: {kernel_size}", flush=True)
       if (stride<1):
           err+=1
           if verbose: dlr_common.DlrError(f"stride should be larger than 0: {stride}", flush=True)
       if (padding<0):
           err+=1
           if verbose: dlr_common.DlrError(f"padding should be positive: {padding}", flush=True)
    if err>0: return False, int(((in_size-kernel_size+2*padding)/stride)+1)
    else:     return True, int(((in_size-kernel_size+2*padding)/stride)+1)

#===============================================================================
def Convolution2d( out_data    # out_channel x out_size x out_size
                 , in_data     # in_channel x in_size x in_size
                 , kernel      # out_channel x in_channel x kernel_size x kernel_size
                 , bias=None   # out_channel
                 , stride=1
                 , padding=0
                 , rigor=False
                 , verbose=False):
    """
    Returns True on success, otherwize returns False
    Applies a 2D convolution over an input data composed of several input channels.
    Note that all nd-array lists are NumPy (mutable), not PyTorch tensor (immutable).
    :param out_data: <mutable> output data, out_data[out_channel][out_size][out_size]
    :param in_data: input data, in_data[in_channel][in_size][in_size]
    :param kernel: kernel (or filter), kernel[out_channel][in_channel][kernel_size][kernel_size]
    :param bias: bias for each filter (kernel), bias[out_channel]
    :param stride: num of skips to apply next filter
    :param padding: num of pixes at the boundary
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: 'True' on success, 'False' on failure.
    Follwoings are derived from input arguments
    . out_size: array size of out_data
    . in_size: array size of in_data
    . kernel_size: dimension of filter, e.g., 3 means 3x3 kernel
    . in_chnannels: num of input channels, e.g., 3 for RGB, 1 for gray
    . out_channels: num of filters
    . bias_size: array size of bias
    Following is an example usage for PyTorch.
        Convolution2d( tensor_out_data.data.numpy() # out_channel x out_size x out_size
                         , tenso_in_data.data.numpy()   # in_channel x in_size x in_size
                         , tensor_kernel.data.numpy()   # in_channel x out_channel x kernel_size x kernel_size
                         , tensor_bias.data.numpy()     # out_channel
                         , stride
                         , padding
                         , rigor=True
                         , verbose=True)
    """
    if rigor or dlr_common.rigor:
       error =0
       if (out_data.ndim!=3):
           error += 1
           if verbose: dlr_common.DlrError("out_data is not 3 dim", flush=True)
       if (in_data.ndim!=3):
           error += 1
           if verbose: dlr_common.DlrError("in_data is not 3 dim", flush=True)
       if (kernel.ndim!=4):
           error += 1
           if verbose: dlr_common.DlrError("kernel is not 4 dim", flush=True)
       if (bias is not None) and (bias.ndim!=1):
           error += 1
           if verbose: dlr_common.DlrError(f"bias should be 1 dim: {bias.ndim}", flush=True)
       if (stride<1):
           error += 1
           if verbose: dlr_common.DlrError(f"stride should be >=1: {stride}", flush=True)
       if (padding<0):
           error += 1
           if verbose: dlr_common.DlrError(f"padding should be >=0: {padding}", flush=True)
       t_out_size    = out_data.shape[2] # note ndim (i.e., rank) is 3
       t_in_size     = in_data.shape[2] # note ndim (i.e., rank) is 3
       t_kernel_size = kernel.shape[3] # note ndim (i.e., rank) is 4
       t_in_channel  = in_data.shape[0]
       t_out_channel = out_data.shape[0]
       t_stride      = stride
       t_padding     = padding
       status, t_out_size_expect = GetOutputSizeOfConvolution2d( in_size=t_in_size
                                                               , kernel_size=t_kernel_size
                                                               , stride=t_stride
                                                               , padding=t_padding
                                                               , rigor=rigor
                                                               , verbose=verbose)
       if not status: return False # something wrong with arguments
       if (t_out_size!=t_out_size_expect):
           error += 1
           dlr_common.DlrError(f"out_size mis-match: {t_out_size, t_out_size_expect}", flush=True)
       if ((t_kernel_size%2)!=1):
           error += 1
           dlr_common.DlrError(f"kernel_size should be odd: {t_kernel_size}", flush=True)
       if verbose:
          dlr_common.DlrInfo(f"out_channel={t_out_channel} {out_data.shape}")
          dlr_common.DlrInfo(f"in_channel ={t_in_channel} {in_data.shape}")
          dlr_common.DlrInfo(f"out_size   ={t_out_size} {out_data.shape}")
          dlr_common.DlrInfo(f"in_size    ={t_in_size} {in_data.shape}")
          dlr_common.DlrInfo(f"kernel_size={t_kernel_size} {kernel.shape}")   
          dlr_common.DlrInfo(f"stride     ={t_stride} {stride}")
          dlr_common.DlrInfo(f"padding    ={t_padding} {padding}")
       if (error!=0):
           dlr_common.DlrError(" parameter mis-match", flush=True)
           return False
    #_fname=''
    #_ctype=''
    if out_data.dtype.type == np.int32:
        _fname = 'Convolution2dInt'
        _ctype = ctypes.c_int
    elif out_data.dtype.type == np.float32:
        _fname = 'Convolution2dFloat'
        _ctype = ctypes.c_float
    elif out_data.dtype.type == np.float64:
        _fname = 'Convolution2dDouble'
        _ctype = ctypes.c_double
    else:
        dlr_common.DlrError(" not support "+str(out_data.dtype.type), flush=True)
        return False
    _Conv2d=dlr_common.WrapFunction(dlr_common._dlr
                            ,_fname
                            , None          # return type
                            ,[ctypes.POINTER(_ctype) # output features
                             ,ctypes.POINTER(_ctype) # input image
                             ,ctypes.POINTER(_ctype) # kernels
                             ,ctypes.POINTER(_ctype)  # bias
                             ,ctypes.c_ushort  # out_size
                             ,ctypes.c_ushort  # in_size
                             ,ctypes.c_ubyte   # kernel_size (only for square filter)
                             ,ctypes.c_ushort  # bias_size
                             ,ctypes.c_ushort  # in_channel
                             ,ctypes.c_ushort  # out_channel
                             ,ctypes.c_ubyte   # stride
                             ,ctypes.c_ubyte   # padding
                             ,ctypes.c_int     # rigor
                             ,ctypes.c_int ])  # verbose
    CP_out_data    = out_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_data     = in_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_kernel      = kernel.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_out_size    = ctypes.c_ushort(out_data.shape[2]) # note ndim (i.e., rank) is 3
    CP_in_size     = ctypes.c_ushort(in_data.shape[2]) # note ndim (i.e., rank) is 3
    CP_kernel_size = ctypes.c_ubyte (kernel.shape[3]) # note ndim (i.e., rank) is 4
    CP_in_channel  = ctypes.c_ushort(in_data.shape[0])
    CP_out_channel = ctypes.c_ushort(kernel.shape[0])
    CP_stride      = ctypes.c_ubyte (stride)
    CP_padding     = ctypes.c_ubyte (padding)
    CP_rigor       = 1 if rigor else 0
    CP_verbose     = 1 if verbose else 0
    if (bias is None) or (bias.size == 0):
       CP_bias        = ctypes.POINTER(_ctype)()
       CP_bias_size   = ctypes.c_ushort(0)
    else:
       CP_bias        = bias.ctypes.data_as(ctypes.POINTER(_ctype))
       CP_bias_size   = ctypes.c_ushort(bias.shape[0])
    _Conv2d(CP_out_data
           ,CP_in_data
           ,CP_kernel
           ,CP_bias
           ,CP_out_size
           ,CP_in_size
           ,CP_kernel_size
           ,CP_bias_size
           ,CP_in_channel
           ,CP_out_channel
           ,CP_stride
           ,CP_padding
           ,CP_rigor
           ,CP_verbose)
    return True

#===============================================================================
if __name__=='__main__':
    def TestConvolution2d(_dtype):
        """
        _dtype: specify data type of data one of {np.int32, np.float32, np.float64}
        """
        in_channel = 2
        in_size = 5
        in_data = np.empty([in_channel,in_size,in_size], dtype=_dtype)
        out_channel = 1
        kernel_size = 3
        kernel = np.empty([out_channel,in_channel,kernel_size,kernel_size], dtype=_dtype)
        bias = None #bias = np.zeros([out_channel], dtype=_dtype)
        stride = 1
        padding = 0
        status, out_size = GetOutputSizeOfConvolution2d( in_size
                                                       , kernel_size
                                                       , stride
                                                       , padding )
        if not status: return
        # prepare arrays to pass to C function
        out_data = np.empty([out_channel,out_size,out_size], dtype=_dtype)
        v = 0
        for i in range(in_channel):
            for r in range(in_size):
                for c in range(in_size):
                    in_data[i][r][c] = v % 10
                    v += 1
        for o in range(out_channel):
            for i in range(in_channel):
                for r in range(kernel_size):
                    for c in range(kernel_size): # make identity matrix
                        if (r==c): kernel[o][i][r][c] = 1
                        else     : kernel[o][i][r][c] = 0
        if bias is not None:
            for b in range(out_channel): bias[b] = 0
    
        status = Convolution2d( out_data    # out_channel x out_size x out_size
                              , in_data     # in_channel x in_size x in_size
                              , kernel      # out_channel x in_channel x kernel_size x kernel_size
                              , bias        # out_channel
                              , stride
                              , padding)
        if status:
            dlr_common.DlrPrint(f"in_data:\n{in_data}", flush=True)
            dlr_common.DlrPrint(f"kernel:\n{kernel}", flush=True)
            dlr_common.DlrPrint(f"bias:\n{bias}", flush=True)
            dlr_common.DlrPrint(f"out_data:\n{out_data}", flush=True)

#===============================================================================
if __name__=='__main__':
    dlr_common.DlrPrint("Testing Convolution2d", flush=True)
    dlr_common.DlrPrint("*********************", flush=True)
    TestConvolution2d(_dtype=np.int32)
    TestConvolution2d(_dtype=np.float32)
    TestConvolution2d(_dtype=np.float64)

#===============================================================================
# Revision history:
#
# 2020.09.30: argument order of bias and bias_size changed
# 2020.04.25: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
