#!/usr/bin/env python
"""
This file contains Python interface of linear_nd.
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
__description__= "Python interface of linear_nd"

#-------------------------------------------------------------------------------
import ctypes
import ctypes.util
import numpy as np
from python.modules import dlr_common

#===============================================================================
def LinearNd( out_data    # ndim x out_size
            , in_data     # ndim x in_size
            , weight      # out_size x in_size
            , bias=None   # out_size
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
        LinearNd( tensor_out_data.data.numpy() # ndim x out_size
                    , tenso_in_data.data.numpy()   # ndim x in_size
                    , tensor_weight.data.numpy()   # out_size x in_size
                    , tensor_bias.data.numpy()     # out_size
                    , rigor=True
                    , verbose=True)
    """
    if rigor:
       error =0
       if (out_data.ndim!=2):
           error += 1
           if verbose: dlr_common.DlrError("out_data is not 1 dim", flush=True)
       if (in_data.ndim!=2):
           error += 1
           if verbose: dlr_common.DlrError("in_data is not 1 dim", flush=True)
       if (weight.ndim!=2):
           error += 1
           if verbose: dlr_common.DlrError("weight is not 2 dim", flush=True)
       if (bias is not None) and (bias.ndim!=1):
           error += 1
           if verbose: dlr_common.DlrError(f"bias should be 1 dim: {bias.ndim}", flush=True)
       t_out_ndim        = out_data.shape[0]
       t_out_size        = out_data.shape[1] # note ndim (i.e., rank) is 1
       t_in_ndim         = in_data.shape[0]
       t_in_size         = in_data.shape[1] # note ndim (i.e., rank) is 1
       t_weight_size_row = weight.shape[0] # note ndim (i.e., rank) is 2
       t_weight_size_col = weight.shape[1] # note ndim (i.e., rank) is 2
       if (t_out_ndim!=t_in_ndim):
           error += 1
           dlr_common.DlrError(f"dimension mis-match", flush=True)
       if (t_out_size!=t_weight_size_row):
           error += 1
           dlr_common.DlrError(f"out_size mis-match", flush=True)
       if (t_in_size!=t_weight_size_col):
           error += 1
           dlr_common.DlrError(f"out_size mis-match", flush=True)
       if verbose:
          dlr_common.DlrInfo(f"out_size   ={t_out_size} {out_data.shape}")
          dlr_common.DlrInfo(f"in_size    ={t_in_size} {in_data.shape}")
          dlr_common.DlrInfo(f"weight_size={t_weitht_dize_row} {t_weight_size_col}")
       if (error!=0):
           dlr_common.DlrError(" parameter mis-match", flush=True)
           return False
    #_fname=''
    #_ctype=''
    if out_data.dtype.type == np.int32:
        _fname = 'LinearNdInt'
        _ctype = ctypes.c_int
    elif out_data.dtype.type == np.float32:
        _fname = 'LinearNdFloat'
        _ctype = ctypes.c_float
    elif out_data.dtype.type == np.float64:
        _fname = 'LinearNdDouble'
        _ctype = ctypes.c_double
    else:
        dlr_common.DlrError(" not support "+str(out_data.dtype.type), flush=True)
        return False
    _LinearNd=dlr_common.WrapFunction(dlr_common._dlr
                              ,_fname
                              , None          # return type
                              ,[ctypes.POINTER(_ctype) # out data
                               ,ctypes.POINTER(_ctype) # in data
                               ,ctypes.POINTER(_ctype) # weight
                               ,ctypes.POINTER(_ctype) # bias
                               ,ctypes.c_ushort  # out_size
                               ,ctypes.c_ushort  # in_size
                               ,ctypes.c_ushort  # bias_size
                               ,ctypes.c_ubyte   # ndim
                               ,ctypes.c_int     # rigor
                               ,ctypes.c_int ])  # verbose
    CP_out_data    = out_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_data     = in_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_weight      = weight.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_out_size    = ctypes.c_ushort(out_data.shape[1]) # note ndim (i.e., rank) is 2
    CP_in_size     = ctypes.c_ushort(in_data.shape[1]) # note ndim (i.e., rank) is 2
    CP_ndim        = ctypes.c_ubyte(in_data.shape[0]) # note ndim (i.e., rank) is 2
    CP_rigor       = 1 if rigor else 0
    CP_verbose     = 1 if verbose else 0
    if (bias is None) or (bias.size == 0):
       CP_bias        = ctypes.POINTER(_ctype)()
       CP_bias_size   = ctypes.c_ushort(0)
    else:
       CP_bias        = bias.ctypes.data_as(ctypes.POINTER(_ctype))
       CP_bias_size   = ctypes.c_ushort(bias.shape[0])
    _LinearNd(CP_out_data
             ,CP_in_data
             ,CP_weight
             ,CP_bias
             ,CP_out_size
             ,CP_in_size
             ,CP_bias_size
             ,CP_ndim
             ,CP_rigor
             ,CP_verbose)
    return True

#===============================================================================
if __name__=='__main__':
    def TestLinearNd(_dtype):
        """
        _dtype: specify data type of data one of {np.int32, np.float32, np.float64}
        """
        ndim    = 2
        in_size = 5
        in_data = np.empty([ndim, in_size], dtype=_dtype)
        out_size = 3
        out_data = np.empty([ndim, out_size], dtype=_dtype)
        weight_size = [out_size, in_size]
        weight = np.empty(weight_size, dtype=_dtype)
        bias = None #bias = np.zeros([out_size], dtype=_dtype)
        v = 0
        for n in range(ndim):
            for i in range(in_size):
                in_data[n][i] = v % 10
                v += 1
        for r in range(out_size):
            for c in range(in_size): # make identity matrix
                if (r==c): weight[r][c] = 1
                else     : weight[r][c] = 0
        if bias is not None:
            for b in range(out_size): bias[b] = 0
    
        status = LinearNd( out_data    # ndim x out_size
                         , in_data     # ndim x in_size
                         , weight      # out_size x in_size
                         , bias        # out_size
                         )
        if status:
            dlr_common.DlrPrint(f"in_data:\n{in_data}", flush=True)
            dlr_common.DlrPrint(f"weight:\n{weight}", flush=True)
            dlr_common.DlrPrint(f"bias:\n{bias}", flush=True)
            dlr_common.DlrPrint(f"out_data:\n{out_data}", flush=True)

#===============================================================================
if __name__=='__main__':
    dlr_common.DlrPrint("Testing LinarNd", flush=True)
    dlr_common.DlrPrint("*********************", flush=True)
    TestLinearNd(_dtype=np.int32)
    TestLinearNd(_dtype=np.float32)
    TestLinearNd(_dtype=np.float64)

#===============================================================================
# Revision history:
#
# 2020.09.30: argument order of bias and bias_size changed
# 2020.04.25: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
