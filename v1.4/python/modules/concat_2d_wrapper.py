#!/usr/bin/env python
"""
This file contains Python interface of concate_2d.
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
__description__= "Python interface of concat_2d"

#-------------------------------------------------------------------------------
import ctypes
import ctypes.util
import numpy as np
import math
from python.modules import dlr_common

#===============================================================================
def Concat2d( out_data    #
            , in_dataA    # in_rowsA x in_colsA
            , in_dataB    # in_rowsB x in_colsB
            , dim=0
            , rigor=False
            , verbose=False):
    """
    Returns True on success, otherwize returns False
    Applies a 2D Concatenation over two 2-dimensional input data
    Note that all nd-array lists are NumPy (mutable), not PyTorch tensor (immutable).
    :param out_data: <mutable> output data, out_data[][]
    :param in_dataA: input data, in_dataA[in_rowsA][in_colsA]
    :param in_dataB: input data, in_dataB[in_rowsB][in_colsB]
    :param dim: dimension to concatenate, 0 or 1
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: 'True' on success, 'False' on failure.
    Follwoings are derived from input arguments
    . out_rows:
    . out_cols:
    . in_rowsA:
    . in_colsA:
    . in_rowsB:
    . in_colsB:
    . dim:
    Following is an example usage for PyTorch.
        Concat2d( tensor_out_data.data.numpy()
                    , tenso_in_dataA.data.numpy()
                    , tenso_in_dataB.data.numpy()
                    , dim
                    , rigor=True
                    , verbose=True)
    """
    if rigor:
       error =0
       if (out_data.ndim!=2):
           error += 1
           if verbose: dlr_common.DlrError("out_data is not 2 dim")
       if (in_dataA.ndim!=2): 
           error += 1
           if verbose: dlr_common.DlrError("in_data is not 2 dim")
       if (in_dataB.ndim!=2): 
           error += 1
           if verbose: dlr_common.DlrError("in_data is not 2 dim")
       if (dim!=0) and (dim!=1):
           error += 1
           if verbose: dlr_common.DlrError("dim should be 0 or 1")
       t_in_rowsA   = in_dataA.shape[0]
       t_in_colsA   = in_dataA.shape[1]
       t_in_rowsB   = in_dataB.shape[0]
       t_in_colsB   = in_dataB.shape[1]
       if dim==0:
          t_out_rows   = in_dataA.shape[0]+in_dataB.shape[0]
          t_out_cols   = in_dataA.shape[1]
       else:
          t_out_rows   = in_dataA.shape[0]
          t_out_cols   = in_dataA.shape[1]+in_dataB.shape[1]
       if (t_out_rows!=out_data.shape[0]):
           error += 1
           if verbose: dlr_common.DlrError("out data row count error")
       if (t_out_cols!=out_data.shape[1]):
           error += 1
           if verbose: dlr_common.DlrError("out data column count error")
       if dim==0:
           if (t_in_colsA!=t_in_colsB):
               error += 1
               if verbose: dlr_common.DlrError("in dimension eror")
       else:
           if (t_in_rowsA!=t_in_rowsB):
               error += 1
               if verbose: dlr_common.DlrError("in dimension eror")
       if verbose:
          dlr_common.DlrInfo(f"out_data={out_data.shape}")
          dlr_common.DlrInfo(f"in_dataA={in_dataA.shape}")
          dlr_common.DlrInfo(f"in_dataB={in_dataB.shape}")
          dlr_common.DlrInfo(f"dim     ={dim}")
       if (error!=0):
           dlr_common.DlrError("parameter mis-match");
           return False
    #_fname=''
    #_ctype=''
    if out_data.dtype.type == np.int32:
        _fname = 'Concat2dInt'
        _ctype = ctypes.c_int
    elif out_data.dtype.type == np.float32:
        _fname = 'Concat2dFloat'
        _ctype = ctypes.c_float
    elif out_data.dtype.type == np.float64:
        _fname = 'Concat2dDouble'
        _ctype = ctypes.c_double
    else:
        dlr_common.DlrError("not support "+str(out_data.dtype.type))
        return False
    _Concat2d=dlr_common.WrapFunction(dlr_common._dlr
                              ,_fname
                              , None          # return type
                              ,[ctypes.POINTER(_ctype) # output
                               ,ctypes.POINTER(_ctype) # input
                               ,ctypes.POINTER(_ctype) # input
                               ,ctypes.c_ushort  # in_rowsA
                               ,ctypes.c_ushort  # in_colsA
                               ,ctypes.c_ushort  # in_rowsB
                               ,ctypes.c_ushort  # in_colsB
                               ,ctypes.c_ubyte   # dim
                               ,ctypes.c_int     # rigor
                               ,ctypes.c_int     # verbose
                               ]) 
    CP_out_data    = out_data.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_dataA    = in_dataA.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_dataB    = in_dataB.ctypes.data_as(ctypes.POINTER(_ctype))
    CP_in_rowsA    = ctypes.c_ushort(in_dataA.shape[0])
    CP_in_colsA    = ctypes.c_ushort(in_dataA.shape[1])
    CP_in_rowsB    = ctypes.c_ushort(in_dataB.shape[0])
    CP_in_colsB    = ctypes.c_ushort(in_dataB.shape[1])
    CP_dim         = ctypes.c_ubyte(dim)
    CP_rigor       = 1 if rigor else 0
    CP_verbose     = 1 if verbose else 0

    _Concat2d(CP_out_data    
             ,CP_in_dataA
             ,CP_in_dataB
             ,CP_in_rowsA
             ,CP_in_colsA
             ,CP_in_rowsB
             ,CP_in_colsB
             ,CP_dim
             ,CP_rigor
             ,CP_verbose
             )
    return True

#===============================================================================
# # Testing function

if __name__=='__main__':
    def TestConcat2d(_dtype):
        """
        _dtype: specify data type of data one of {np.int32, np.float32, np.float64}
        """
        dims=[0,1]
        for dim in dims:
            if dim==0:
               in_rowsA=3
               in_colsA=4
               in_rowsB=5
               in_colsB=in_colsA
               out_rows=in_rowsA+in_rowsB
               out_cols=in_colsA
            else:
               in_rowsA=3
               in_colsA=4
               in_rowsB=in_rowsA
               in_colsB=4
               out_rows=in_rowsA
               out_cols=in_colsA+in_colsB
            in_dataA = np.empty([in_rowsA,in_colsA], dtype=_dtype)
            in_dataB = np.empty([in_rowsB,in_colsB], dtype=_dtype)
            out_data = np.empty([out_rows,out_cols], dtype=_dtype)
            v = 0
            for r in range(in_rowsA):
                for c in range(in_colsA):
                    in_dataA[r][c] = v    
                    v += 1
            for r in range(in_rowsB):
                for c in range(in_colsB):
                    in_dataB[r][c] = v    
                    v -= 1

            print(f"in_dataA={in_dataA.shape}");
            print(f"in_dataB={in_dataB.shape}");
            print(f"out_data={out_data.shape}");
            status = Concat2d( out_data
                             , in_dataA
                             , in_dataB
                             , dim
                             , rigor=True
                             , verbose=True)
            if status:
                dlr_common.DlrPrint(f"in_dataA:\n{in_dataA}")
                dlr_common.DlrPrint(f"in_dataB:\n{in_dataB}")
                dlr_common.DlrPrint(f"out_data:\n{out_data}")

if __name__=='__main__':
    dlr_common.DlrPrint("Testing Conat2d", flush=True);
    dlr_common.DlrPrint("*********************", flush=True)
    TestConcat2d(_dtype=np.int32)
    #TestConcat2d(_dtype=np.float32)
    #TestConcat2d(_dtype=np.float64)

#===============================================================================
# Revision history:
#
# 2020.04.58: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
