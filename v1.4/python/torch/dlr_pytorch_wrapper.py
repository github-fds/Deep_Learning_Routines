#!/usr/bin/env python
"""
This file contains Python interface of convolution_2d.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#-------------------------------------------------------------------------------
__author__     = "Ando Ki"
__copyright__  = "Copyright 2020, Future Design Systems"
__credits__    = ["none", "some"]
__license__    = "FUTURE DESIGN SYSTEMS SOFTWARE END-USER LICENSE AGREEMENT"
__version__    = "0"
__revision__   = "1"
__maintainer__ = "Ando Ki"
__email__      = "contact@future-ds.com"
__status__     = "Development"
__date__       = "2020.09.30"
__description__= "PyTorch interface of Deep Learning Processing Routines"

#-------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import python.modules as _dlr

#===============================================================================
def conv2d( input     # in_minibatch x in_channel x in_size x in_size
          , weight    # out_channel  x in_channel x kernel_size x kernel_size
          , bias=None # out_channel
          , stride=1
          , padding=0
          , dilation=1
          , groups=1
          , rigor=False
          , verbose=False):
    """
    Corresponding torch.nn.functional.conv2d(input, weight, bias=None,
                                             stride, padding, dilation, groups)
    Returns output tensor on success
    Applies a 2D convolution over an input data composed of several input channels.
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_channel][in_size][in_size]
    :param weight: kernel (or filter), weight[out_channel][in_channel][kernel_size][kernel_size]
    :param bias: bias for each filter (kernel), bias[out_channel]
    :param stride: num of skips to apply next filter
    :param padding: num of pixes at the boundary
    :param dilation:
    :param groups:
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if (dilation!=1): error += 1 # not support
       if (groups!=1): error += 1 # not support
       if (input.dim()!=4): error += 1
       if (input.shape[2]!=input.shape[3]): error += 1 # not square
       if (weight.dim()!=4): error += 1
       if (weight.shape[2]!=weight.shape[3]): error += 1 # not square
       if (bias is not None) and (bias.dim()!=1): error += 1
       if (input.shape[1]!=weight.shape[1]): error += 1 # in_channel
       if (bias is not None) and (bias.shape[0]!=weight.shape[0]): error += 1 # out_channel
       if (stride<=0) or (padding<0): error += 1
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    in_size = input.shape[3]
    kernel_size = weight.shape[3]
    out_channel = weight.shape[0]
    status, out_size = _dlr.GetOutputSizeOfConvolution2d( in_size
                                                        , kernel_size
                                                        , stride
                                                        , padding
                                                        , rigor=rigor
                                                        , verbose=verbose)
    if not status: return None
    out_data = torch.empty([in_minibatch,out_channel,out_size,out_size], dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Convolution2d( xout_data.data.numpy() # out_channel x out_size x out_size
                                   , xin_data.data.numpy()  # in_channel x in_size x in_size
                                   , weight.data.numpy()   # in_channel x out_channel x kernel_size x kernel_size
                                   , bias.data.numpy() if bias is not None else None
                                   , stride
                                   , padding
                                   , rigor=rigor
                                   , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
def max_pool2d ( input     # in_minibatch x in_channel x in_size x in_size
               , kernel_size
               , stride=1
               , padding=0
               , ceil_mode=False
               , rigor=False
               , verbose=False):
    """
    Corresponding torch.nn.functional.max_pool2d(input, kernel_size,
                                                 stride, padding, ceil_mode,
                                                 count_include_pad=True,
                                                 divisor_override=None)
    Returns output tensor on success
    Applies a 2D max pooling over an input data composed of several input channels.
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_channel][in_size][in_size]
    :param kernel_size: size of kernel
    :param stride: num of skips to apply next filter
    :param padding: num of pixes at the boundary
    :param ceil_mode: when True, will use ceil instead of floor in the formula to compute the output shape
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if ceil_mode: error += 1 # not support
       if (input.dim()!=4): error += 1
       if (input.shape[2]!=input.shape[3]): error += 1 # not square
       if (kernel_size<=0): error += 1
       if (stride<=0) or (padding<0): error += 1
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    in_size = input.shape[3]
    out_channel = input.shape[1]
    status, out_size = _dlr.GetOutputSizeOfPooling2dMax( in_size
                                                       , kernel_size
                                                       , stride
                                                       , padding
                                                       , ceil_mode
                                                       , rigor=rigor
                                                       , verbose=verbose)
    if not status: return None
    out_data = torch.empty([in_minibatch,out_channel,out_size,out_size], dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Pooling2dMax( xout_data.data.numpy() # out_channel x out_size x out_size
                                 , xin_data.data.numpy()  # in_channel x in_size x in_size
                                 , kernel_size
                                 , stride
                                 , padding
                                 , ceil_mode
                                 , rigor=rigor
                                 , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
def avg_pool2d ( input     # in_minibatch x in_channel x in_size x in_size
               , kernel_size
               , stride=1
               , padding=0
               , ceil_mode=False
               , rigor=False
               , verbose=False):
    """
    Corresponding torch.nn.functional.avg_pool2d(input, kernel_size,
                                                 stride, padding, ceil_mode,
                                                 count_include_pad=True,
                                                 divisor_override=None)
    Returns output tensor on success
    Applies a 2D average pooling over an input data composed of several input channels.
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_channel][in_size][in_size]
    :param kernel_size: size of kernel
    :param stride: num of skips to apply next filter
    :param padding: num of pixes at the boundary
    :param ceil_mode: when True, will use ceil instead of floor in the formula to compute the output shape
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if ceil_mode: error += 1 # not support
       if (input.dim()!=4): error += 1
       if (input.shape[2]!=input.shape[3]): error += 1 # not square
       if (kernel_size<=0): error += 1
       if (stride<=0) or (padding<0): error += 1
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    in_size = input.shape[3]
    out_channel = input.shape[1]
    status, out_size = _dlr.GetOutputSizeOfPooling2dAvg( in_size
                                                      , kernel_size
                                                      , stride
                                                      , padding
                                                      , ceil_mode
                                                      , rigor=rigor
                                                      , verbose=verbose)
    if not status: return None
    out_data = torch.empty([in_minibatch,out_channel,out_size,out_size], dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Pooling2dAvg( xout_data.data.numpy() # out_channel x out_size x out_size
                                 , xin_data.data.numpy()  # in_channel x in_size x in_size
                                 , kernel_size
                                 , stride
                                 , padding
                                 , ceil_mode
                                 , rigor=rigor
                                 , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
# example: def __init__(self):
#              super().__init__()
#              self.a1 = nn.Linear(4,4)
#              self.a2 = nn.Linear(4,4)
#              self.a3 = nn.Linear(9,1)
# example: def forward(self,x):
#              o1 = self.a1(x)
#              o2 = self.a2(x).transpose(1,2)
#              output = torch.bmm(o1,o2)
#              output = output.view(len(x),9)
#              output = self.a3(output)
#              return output
# 
# Linear layer accept only 1D input ==> so linearNd() may be removed.
def linear ( input   # in_minibatch x in_size
           , weight  # out_size x in_size
           , bias=None   # out_size
           , rigor=False
           , verbose=False):
    """
    Correspond torch.nn.functional.linear(input, weight, bias)
    Returns output tensor on success
    Applies a 1D vector matrix multiplication over an input data
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][1][in_size]
    :param weight: weight[out_size][in_size]
    :param bias: bias[out_size]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if (input.dim()==2):
        return linear1d( input   # in_minibatch x N x in_size
                       , weight  # out_size x in_size
                       , bias
                       , rigor
                       , verbose)
    else:
        return linearNd( input   # in_minibatch x N x in_size
                       , weight  # out_size x in_size
                       , bias
                       , rigor
                       , verbose)

#===============================================================================
# Z = X * W' + B, where W' is transposed
def linear1d ( input   # in_minibatch x in_size
             , weight  # out_size x in_size
             , bias=None   # out_size
             , rigor=False
             , verbose=False):
    """
    Correspond torch.nn.functional.linear(input, weight, bias)
    Returns output tensor on success
    Applies a 1D vector matrix multiplication over an input data
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_size]
    :param weight: weight[out_size][in_size]
    :param bias: bias[out_size]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if (input.dim()!=2): error += 1
       if (weight.dim()!=2): error += 1 # not 2D
       if (weight.shape[1]!=input.shape[1]): error += 1
       if (bias is not None) and (bias.dim()!=1): error += 1
       if (bias is not None) and (bias.shape[0]!=weight.shape[0]): error += 1
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    in_size = input.shape[1]
    out_size = weight.shape[0]
    out_data = torch.empty([in_minibatch,out_size], dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Linear1d( xout_data.data.numpy() # out_size
                              , xin_data.data.numpy()  # in_size
                              , weight.data.numpy() # out_size x in_size
                              , None if bias is None else bias.data.numpy() # out_size
                              , rigor=rigor
                              , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
# Z = X * W' + B, where W' is transposed
def linearNd ( input   # in_minibatch x in_size x ...
             , weight  # out_size x in_size
             , bias=None   # out_size
             , rigor=False
             , verbose=False):
    """
    Correspond torch.nn.functional.linear(input, weight, bias)
    Returns output tensor on success
    Applies a N-D vector matrix multiplication over an input data
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_size][in_size][in_size]...
    :param weight: weight[out_size][in_size]
    :param bias: bias[out_size]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if (input.dim()==2):
        return linear1d( input   # in_minibatch x N x in_size
                       , weight  # out_size x in_size
                       , bias
                       , rigor
                       , verbose)
    if rigor:
       error = 0
       if (weight.dim()!=2): error += 1 # not 2D
       if (weight.shape[1]!=input.shape[2]): error += 1
       if (bias is not None) and (bias.dim()!=1): error += 1
       if (bias is not None) and (bias.shape[0]!=weight.shape[0]): error += 1
       if error!=0: return None
    in_minibatch = input.shape[0]
    out_data = torch.empty([in_minibatch,input.shape[0],weight.shape[1]], dtype=input.dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.LinearNd( xout_data.data.numpy() # ndim x out_size
                              , xin_data.data.numpy()  # ndim x in_size
                              , weight.data.numpy() # out_size x in_size
                              , None if bias is None else bias.data.numpy() # out_size
                              , rigor=rigor
                              , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
def cat( tensors
       , dim=0
       , rigor=False
       , verbose=False):
    """
    Correspond torch.cat(tensors,dim,out=None) for tensor.dim is 3, i.e, (minibatch,rows,cols)
    """
    if (tensors.numel()!=2): return None
    return concat2d( tensors[0]
                   , tensors[1]
                   , dim
                   , rigor
                   , verbose)

#===============================================================================
def concat2d( inputA # minibatch x rowsA x colsA
            , inputB # minibatch x rowsB x colsB
            , dim=0
            , rigor=False
            , verbose=False):
    """
    Correspond torch.cat(tensors,dim,out=None) for tensor.dim is 3, i.e, (minibatch,rows,cols)
    Returns output tensor on success
    Applies two 2-dimentional concatenation
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param inputA: input data, input[rowsA][colsA]
    :param inputB: input data, input[rowsB][colsB]
    :param dim: dimension
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if (inputA.dim()!=3): error += 1
       if (inputB.dim()!=3): error += 1
       if (inputA.shape[0]!=inputB.shape[0]): error += 1 # minibatch
       if (dim!=0) and (dim!=1): error += 1
       if error!=0: return None
    if dim==0:
       out_rows = inputA.shape[1]
       out_cols = inputA.shape[2]+inputB.shape[2]
    else:
       out_rows = inputA.shape[1]+inputB.shape[1]
       out_cols = inputA.shape[2]
    dtype = input.dtype
    minibatch = inputA.shape[0]
    out_data = torch.empty([minibatch,out_rows,our_cols], dtype=dtype)
    for mb in range(in_minibatch):
       xout_data  = out_data[mb]
       xin_dataA  = inputA[mb]
       xin_dataB  = inputB[mb]
       status = _dlr.Concat2d( xout_data.data.numpy()
                            , xin_dataA.data.numpy()
                            , xin_dataB.data.numpy()
                            , dim
                            , rigor=rigor
                            , verbose=verbose)
       if not status: return None
       out_data[mb] = xout_data
    return out_data

#===============================================================================
def activations( func
               , input
               , negative_slope=0.01
               , rigor=False
               , verbose=False):
    """
    Bridge to a specific non-linear activation function
    Returns output tensor on success
    Applies activation function
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param func_name: activation function; ReLu, LeakyReLu, Tanh, Sigmoid
    :param input: input data, input[minibatch][....] in any dimension
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    func_name = 'Activation'+func
    function  = getattr(_dlr, func_name)
    minibatch = input.shape[0]
    dtype     = input.dtype
    out_data = torch.empty(input.shape, dtype=dtype)
    for mb in range(minibatch):
       xout_data = out_data[mb]
       xin_data  = input[mb]
       if func == 'LeakyReLu':
           status = function( xout_data.data.numpy()
                            , xin_data.data.numpy()
                            , negative_slope=negative_slope
                            , rigor=rigor
                            , verbose=verbose)
       else:
           # status = _dlr.__getattribute__(func_name)( xout_data.data.numpy()
           status = function( xout_data.data.numpy()
                            , xin_data.data.numpy()
                            , rigor=rigor
                            , verbose=verbose)
       if not status: return None
       out_data[mb] = xout_data
    return out_data

def relu(input, rigor=False, verbose=False):
    """
    Correspond torch.nn.functional.relu(input, inplace=False)
    """
    return activations( 'ReLu'
                      , input
                      , rigor
                      , verbose)
def leaky_relu(input, negative_slope=0.01, rigor=False, verbose=False):
    return activations( 'LeakyReLu'
                      , input
                      , negative_slope
                      , rigor
                      , verbose)
def tanh(input, rigor=False, verbose=False):
    return activations( 'Tanh'
                      , input
                      , rigor
                      , verbose)
def sigmoid(input, rigor=False, verbose=False):
    return activations( 'Sigmoid'
                      , input
                      , rigor
                      , verbose)

#===============================================================================
def batch_norm ( input   # in_minibatch x in_channel x <...>
               , running_mean
               , running_var
               , weight=None
               , bias=None
               , eps=1E-5
               , rigor=False
               , verbose=False):
    """
    Correspond torch.nn.functional.batch_norm(input, running_mean, running_var,
                                              weight, bias,
                                              training=False, momentum=0.1, eps)
    """
    if (input.dim()==3):
        return batch_norm1d(input, running_mean, running_var,
                           weight, bias, eps, rigor, verbose)
    elif (input.dim()==4):
        return batch_norm2d(input, running_mean, running_var,
                           weight, bias, eps, rigor, verbose)
    elif (input.dim()==5):
        return batch_norm3d(input, running_mean, running_var,
                           weight, bias, eps, rigor, verbose)
    else:
        if verbose: _dlr.DlrError(f"batch_norm for more than 3D not supported")
        return None

#===============================================================================
def batch_norm1d ( input   # in_minibatch x 1 x in_size
                 , running_mean # 1 x in_size (not no minibatch)
                 , running_var # 1 x in_size (not no minibatch)
                 , weight=None # 1 x in_size
                 , bias=None # 1 x in_size
                 , eps=1E-5
                 , rigor=False
                 , verbose=False):
    """
    Correspond torch.nn.functional.batch_norm(input, running_mean, running_var,
                                              weight, bias,
                                              training=False, momentum=0.1, eps)
    Returns output tensor on success
    Applies a batch normalization over an input data
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_channel][in_size]
    :param running_mean: running_mean[in_channel]
    :param running_var: running_var[in_channel]
    :param weight: None or weight[in_channel]
    :param bias: None or bias[in_channel]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if (input.dim()!=3) and (input.dim()!=2): error += 1
       if (input.dim()==3):
           in_channel = input.shape[1]
           if (input.dim()!=3): error += 1
           if (running_mean.dim()!=1): error += 1 # mind channel
           if (running_var.dim()!=1): error += 1 # mind channel
           if (running_mean.numel()!=in_channel): error += 1
           if (running_var.numel()!=in_channel): error += 1
           if (weight is not None) and (weight.dim()!=1): error += 1
           if (weight is not None) and (weight.numel()!=in_channel): error += 1
           if (bias is not None) and (bias.dim()!=1): error += 1
           if (bias is not None) and (bias.numel()!=in_channel): error += 1
       else: error += 1; _dlr.DlrError("only supported for data with channel")
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    out_data = torch.empty(input.shape, dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Norm1dBatch( xout_data.data.numpy() # ndim x out_size
                                 , xin_data.data.numpy()  # ndim x in_size
                                 , running_mean.data.numpy() # out_size x in_size
                                 , running_var.data.numpy() # out_size x in_size
                                 , None if weight is None else weight.data.numpy() # out_size
                                 , None if bias is None else bias.data.numpy() # out_size
                                 , eps
                                 , rigor=rigor
                                 , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
def batch_norm2d ( input   # in_minibatch x in_channel x in_size x in_size
                 , running_mean # in_channel
                 , running_var # in_channel
                 , weight=None # in_channel x in_size x in_size
                 , bias=None # in_channel x in_size x in_size
                 , eps=1E-5
                 , rigor=False
                 , verbose=False):
    """
    Correspond torch.nn.functional.batch_norm(input, running_mean, running_var,
                                              weight, bias,
                                              training=False, momentum=0.1, eps)
    Returns output tensor on success
    Applies a batch normalization over an input data
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_channel][in_size][in_size]
    :param running_mean: running_mean[in_channel]
    :param running_var: running_var[in_channel]
    :param weight: None or weight[in_channel]
    :param bias: None or bias[in_channel]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if (input.dim()!=4): error += 1
       in_channel = input.shape[1]
       if (running_mean.dim()!=1): error += 1 # mind channel
       if (running_var.dim()!=1): error += 1 # mind channel
       if (running_mean.numel()!=in_channel): error += 1 # not 2D
       if (running_var.numel()!=in_channel): error += 1 # not 2D
       if (weight is not None) and (weight.dim()!=1): error += 1
       if (weight is not None) and (weight.numel()!=in_channel): error += 1
       if (bias is not None) and (bias.dim()!=1): error += 1
       if (bias is not None) and (bias.numel()!=in_channel): error += 1
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    out_data = torch.empty(input.shape, dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Norm2dBatch( xout_data.data.numpy() # ndim x out_size
                                 , xin_data.data.numpy()  # ndim x in_size
                                 , running_mean.data.numpy() # out_size x in_size
                                 , running_var.data.numpy() # out_size x in_size
                                 , None if weight is None else weight.data.numpy() # out_size
                                 , None if bias is None else bias.data.numpy() # out_size
                                 , eps
                                 , rigor=rigor
                                 , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
# not fully tested
def batch_norm3d ( input   # in_minibatch x in_channel x in_depth x in_height x in_width
                 , running_mean # in_channel
                 , running_var # in_channel
                 , weight=None # in_channel x in_depth x in_height x in_width
                 , bias=None # in_channel x in_depth x in_height x in_width
                 , eps=1E-5
                 , rigor=False
                 , verbose=False):
    """
    Correspond torch.nn.functional.batch_norm(input, running_mean, running_var,
                                             weight, bias,
                                             training=False, momentum=0.1, eps)
    Returns output tensor on success
    Applies a batch normalization over an input data
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_channel][...]
    :param running_mean: running_mean[in_channel]
    :param running_var: running_var[in_channel]
    :param weight: None or weight[in_channel]
    :param bias: None or bias[in_channel]
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if (input.dim()!=5): error += 1
       in_channel = input.shape[1]
       in_depth   = input.shape[2]
       in_height  = input.shape[3]
       in_width   = input.shape[4]
       if (running_mean.dim()!=1): error += 1 # not 2D
       if (running_var.dim()!=1): error += 1 # not 2D
       if (running_mean.numel()!=in_channel): error += 1 # not 2D
       if (running_var.numel()!=in_channel): error += 1 # not 2D
       if (weight is not None) and (weight.dim()!=1): error += 1
       if (weight is not None) and (weight.numel()!=in_channel): error += 1
       if (bias is not None) and (bias.dim()!=1): error += 1
       if (bias is not None) and (bias.numel()!=in_channel): error += 1
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    out_data = torch.empty(input.shape, dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Norm3dBatch( xout_data.data.numpy() # ndim x out_size
                                 , xin_data.data.numpy()  # ndim x in_size
                                 , running_mean.data.numpy() # out_size x in_size
                                 , running_var.data.numpy() # out_size x in_size
                                 , None if weight is None else weight.data.numpy() # out_size
                                 , None if bias is None else bias.data.numpy() # out_size
                                 , eps
                                 , rigor=rigor
                                 , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
def conv_transpose2d( input     # in_minibatch x in_channel x in_size x in_size
                    , weight    # in_channel  x out_channel x kernel_size x kernel_size
                    , bias=None # out_channel
                    , stride=1
                    , padding=0
                    , output_padding=0
                    , groups=1
                    , dilation=1
                    , rigor=False
                    , verbose=False):
    """
    Corresponding torch.nn.functional.conv_transpose2d(input, weight, bias=None,
                                             stride, padding, dilation, groups)
    Returns output tensor on success
    Applies a 2D deconvolution over an input data composed of several input channels.
    Note that all nd-array lists are PyTorch tensor (immutable).
    :param input: input data, input[in_minibatch][in_channel][in_size][in_size]
    :param weight: kernel (or filter), weight[out_channel][in_channel][kernel_size][kernel_size]
    :param bias: bias for each filter (kernel), bias[out_channel]
    :param stride: num of skips to apply next filter
    :param padding: num of pixes at the boundary
    :param dilation:
    :param groups:
    :param rigor: check values rigorously when 'True'
    :param verbose: output message more when 'True'
    :return: out_data on success, None on failure.
    """
    if rigor:
       error = 0
       if (dilation!=1): error += 1 # not support
       if (groups!=1): error += 1 # not support
       if (output_padding!=0): error += 1 # not support
       if (input.dim()!=4): error += 1
       if (input.shape[2]!=input.shape[3]): error += 1 # not square
       if (weight.dim()!=4): error += 1
       if (weight.shape[2]!=weight.shape[3]): error += 1 # not square
       if (bias is not None) and (bias.dim()!=1): error += 1
       if (input.shape[1]!=weight.shape[0]): error += 1 # in_channel
       if (bias is not None) and (bias.shape[0]!=weight.shape[1]): error += 1 # out_channel
       if (stride<=0) or (padding<0): error += 1
       if error!=0: return None
    dtype = input.dtype
    in_minibatch = input.shape[0]
    in_channel = input.shape[1]
    in_size = input.shape[3]
    kernel_size = weight.shape[3]
    out_channel = weight.shape[1]
    status, out_size = _dlr.GetOutputSizeOfDeconvolution2d( in_size=in_size
                                                          , kernel_size=kernel_size
                                                          , stride=stride
                                                          , padding=padding
                                                          , output_padding=0
                                                          , dilation=1
                                                          , rigor=rigor
                                                          , verbose=verbose)
    if not status: return None
    out_data = torch.empty([in_minibatch,out_channel,out_size,out_size], dtype=dtype)
    for mb in range(in_minibatch):
        xout_data = out_data[mb]
        xin_data  = input[mb]
        status = _dlr.Deconvolution2d( xout_data.data.numpy() # out_channel x out_size x out_size
                                     , xin_data.data.numpy()  # in_channel x in_size x in_size
                                     , weight.data.numpy()   # in_channel x out_channel x kernel_size x kernel_size
                                     , bias.data.numpy()     # out_channel
                                     , stride
                                     , padding
                                     , rigor=rigor
                                     , verbose=verbose)
        if not status: return None
        out_data[mb] = xout_data
    return out_data

#===============================================================================
if __name__=='__main__':
    def TestLinear1d     (dtype,limit,random,rigor,verbose): return False
    def TestConcat2d     (dtype,limit,random,rigor,verbose): return False
    def TestNorm1dBatch  (dtype,limit,random,rigor,verbose): return False

#===============================================================================
if __name__=='__main__':
    def TestLinearNd     (dtype,limit,random,rigor,verbose):
        return False

#===============================================================================
if __name__=='__main__':
    def TestPooling2dMax(dtype,limit,random,rigor,verbose):
        TestPooling2d(func='max'
                     ,dtype=dtype
                     ,limit=limit
                     ,random=random
                     ,rigor=rigor
                     ,verbose=verbose)

    def TestPooling2dAvg(dtype,limit,random,rigor,verbose):
        TestPooling2d(func='avg'
                     ,dtype=dtype
                     ,limit=limit
                     ,random=random
                     ,rigor=rigor
                     ,verbose=verbose)

    def TestPooling2d(func='max'
                     ,dtype=torch.float32
                     ,limit=1.0E-3 # error limit
                     ,random=False
                     ,rigor=False
                     ,verbose=False):
        configs = [
                   [1, 2,  4,2,1,0,0] #minibatch[0],in_channel[1],in_size[2],kernel_size[3],stride[4],padding[5],ceil[6]
                  ,[1,16,416,2,1,0,0] #minibatch[0],in_channel[1],in_size[2],kernel_size[3],stride[4],padding[5],ceil[6]
                  ,[1,16,416,2,1,1,0] #minibatch[0],in_channel[1],in_size[2],kernel_size[3],stride[4],padding[5],ceil[6]
                  ,[1,256,26,2,2,0,0]
                  ,[1,256,26,2,2,1,0]
                  ,[1,512,13,2,1,0,0]
                  ,[1,512,12,6,1,0,0]
                  ,[1,512,12,6,2,1,0]
                  ,[1,512,12,6,3,2,0]
                  ]
        errors = torch.zeros(len(configs))
        for idx in range(len(configs)):
            minibatch   = configs[idx][0]
            in_channel  = configs[idx][1]
            in_size     = configs[idx][2]
            kernel_size = configs[idx][3] # make it even
            stride      = configs[idx][4]
            padding     = configs[idx][5]
            ceil_mode   = False if configs[idx][6] == 0 else True
            data        = torch.zeros(size=[minibatch,in_channel,in_size,in_size])
            in_data     = GenDataPooling2d(data, rigor=rigor, verbose=verbose)

            sys.stdout.flush()
            if func == 'max':
                out_data  = F.max_pool2d( input=in_data
                                        , kernel_size=kernel_size
                                        , stride=stride
                                        , padding=padding
                                        , ceil_mode=ceil_mode)
                nout_data = max_pool2d ( input=in_data
                                       , kernel_size=kernel_size
                                       , stride=stride
                                       , padding=padding
                                       , ceil_mode=ceil_mode
                                       , rigor=rigor
                                       , verbose=verbose)
            elif func == 'avg':
                out_data  = F.avg_pool2d( input=in_data
                                        , kernel_size=kernel_size
                                        , stride=stride
                                        , padding=padding
                                        , ceil_mode=ceil_mode)
                nout_data = avg_pool2d ( input=in_data
                                       , kernel_size=kernel_size
                                       , stride=stride
                                       , padding=padding
                                       , ceil_mode=ceil_mode
                                       , rigor=rigor
                                       , verbose=verbose)
            else:
                return False

            diff = []
            status = False
            if (out_data is not None) and (nout_data is not None):
                diff = torch.lt(torch.abs(torch.add(out_data, -nout_data)), limit)
                status = torch.all(diff)
                if not status:
                    diff_max = torch.max(torch.abs(torch.add(out_data, -nout_data)))
                    _dlr.DlrWarn(f"diff max: {diff_max}")

            ok = 0; error = 0
            if status:
               ok += 1
               _dlr.DlrInfo(f"OK {out_data.shape}")
               if verbose:
                   _dlr.DlrInfo(f"in_data\n{in_data}")
                   _dlr.DlrInfo(f"kernel_size\n{kernel_size}")
                   _dlr.DlrInfo(f"out_data\n{out_data}")
                   _dlr.DlrInfo(f"nout_data\n{nout_data}")
            else:
               error += 1
               _dlr.DlrError(f"Mis-match {torch.sum(diff==False)} of tensor {out_data.shape}")
               if verbose:
                   _dlr.DlrInfo(f"in_data\n{in_data}")
                   _dlr.DlrInfo(f"kernel_size\n{kernel_size}")
                   _dlr.DlrInfo(f"out_data\n{out_data}")
                   _dlr.DlrInfo(f"nout_data\n{nout_data}")

            errors[idx] = error

        return True if torch.sum(errors)==0.0 else False

    def GenDataPooling2d(data, rigor=False, verbose=False):
        if (data.dim()==4): # minibatch x channel x size x size
            in_minibatch = data.shape[0]
            in_channel   = data.shape[1]
            in_size      = data.shape[2]
        else: return None
        error  = 0

        in_data   = (100+100)*torch.rand(size=data.shape) - 100

        return in_data

#===============================================================================
if __name__=='__main__':
    def TestConvolution2d(dtype=torch.float32
                         ,limit=1.0E-3 # error limit
                         ,random=False
                         ,rigor=False
                         ,verbose=False):
        configs = [
                   [1, 3,416,16,5,1,0]#minibatch,in_chan,in_sizd,out_chan,kerne_size,strid,padding
                  ,[1, 3,416,16,5,2,0]
                  ,[1, 3,416,16,5,3,0]
                  ,[1, 3,416,16,5,1,1]
                  ,[1, 3,416,16,5,2,2]
                  ,[1, 3,416,16,5,3,1]
                  ,[1, 8,416,64,5,1,0]
                  ,[1, 8,416,64,5,1,1]
                  ,[1, 8,416,64,5,1,2]
                  ]

        errors = torch.zeros(len(configs))

        for idx in range(len(configs)):
            minibatch   = configs[idx][0]
            in_channel  = configs[idx][1]
            in_size     = configs[idx][2]
            out_channel = configs[idx][3]
            kernel_size = configs[idx][4]
            stride      = configs[idx][5]
            padding     = configs[idx][6]
            data   = torch.zeros(size=[minibatch,in_channel,in_size,in_size])
            kernel = torch.zeros(size=[out_channel,in_channel,kernel_size,kernel_size])
            bias   = torch.zeros(size=[out_channel])
            in_data,in_kernel,in_bias = GenDataConv2d(data, kernel, bias, rigor=rigor, verbose=verbose)

            sys.stdout.flush()
            out_data  = F.conv2d( input=in_data
                                , weight=in_kernel
                                , bias=in_bias
                                , stride=stride
                                , padding=padding
                                , groups=1
                                , dilation=1)
            sys.stdout.flush()
            nout_data = conv2d ( input=in_data
                               , weight=in_kernel
                               , bias=in_bias
                               , stride=stride
                               , padding=padding
                               , rigor=rigor
                               , verbose=verbose)
            sys.stdout.flush()
            diff = []
            status = False
            if (out_data is not None) and (nout_data is not None):
                diff = torch.lt(torch.abs(torch.add(out_data, -nout_data)), limit)
                status = torch.all(diff)
                if not status:
                    diff_max = torch.max(torch.abs(torch.add(out_data, -nout_data)))
                    _dlr.DlrWarn(f"diff max: {diff_max}")

            ok = 0; error = 0
            if status:
               ok += 1
               _dlr.DlrInfo(f"OK {out_data.shape}")
               if verbose:
                   _dlr.DlrInfo(f"in_data\n{in_data}")
                   _dlr.DlrInfo(f"in_kernel\n{in_kernel}")
                   _dlr.DlrInfo(f"out_data\n{out_data}")
                   _dlr.DlrInfo(f"nout_data\n{nout_data}")
            else:
               error += 1
               _dlr.DlrError(f"Mis-match {torch.sum(diff==False)} of tensor {out_data.shape}")
               if verbose:
                   _dlr.DlrInfo(f"in_data\n{in_data}")
                   _dlr.DlrInfo(f"in_kernel\n{in_kernel}")
                   _dlr.DlrInfo(f"out_data\n{out_data}")
                   _dlr.DlrInfo(f"nout_data\n{nout_data}")
            errors[idx] = error

        return True if torch.sum(errors)==0.0 else False

    def GenDataConv2d(data, kernel, bias, rigor=False, verbose=False):
        if (data.dim()==4): # minibatch x channel x size x size
            in_minibatch = data.shape[0]
            in_channel   = data.shape[1]
            in_size      = data.shape[2]
        else: return None
        out_channel = kernel.shape[1]
        kerne_size  = kernel.shape[2]
        bias_size   = bias.shape[0]
        error  = 0
        if (in_channel!=kernel.shape[0]): error += 1
        if (bias is not None) and (out_channel!=bias_size): error += 1

        in_data   = (100+100)*torch.rand(size=data.shape) - 100
        in_kernel = ( 10+ 10)*torch.rand(size=kernel.shape) - 10
        if (bias is not None): in_bias = 5+5*torch.rand(size=bias.shape) - 5
        else: in_bias = None

        return in_data, in_kernel, in_bias

#===============================================================================
if __name__=='__main__':
    def TestDeconvolution2d(dtype=torch.float32
                           ,limit=1.0E-3 # error limit
                           ,random=False
                           ,rigor=False
                           ,verbose=False):
        minibatch   = 1
        in_channel  = 16
        in_size     = 416
        out_channel = 32
        kernel_size = 5
        data   = torch.zeros(size=[minibatch,in_channel,in_size,in_size])
        kernel = torch.zeros(size=[in_channel,out_channel,kernel_size,kernel_size])
        bias   = torch.zeros(size=[out_channel])
        in_data,in_kernel,in_bias = GenDataDeconv2d(data, kernel, bias, rigor=rigor, verbose=verbose)
        stride=1
        padding=0

        sys.stdout.flush()
        out_data  = F.conv_transpose2d( input=in_data
                                      , weight=in_kernel
                                      , bias=in_bias
                                      , stride=stride
                                      , padding=padding
                                      , output_padding=0
                                      , groups=1
                                      , dilation=1)
        sys.stdout.flush()
        nout_data = conv_transpose2d ( input=in_data
                                      , weight=in_kernel
                                      , bias=in_bias
                                      , stride=stride
                                      , padding=padding
                                      , rigor=rigor
                                      , verbose=verbose)
        sys.stdout.flush()
        status = False
        if (out_data is not None) and (nout_data is not None):
            status = torch.all(torch.lt(torch.abs(torch.add(out_data, -nout_data)), limit))
            if not status:
                diff_max = torch.max(torch.abs(torch.add(out_data, -nout_data)))
                _dlr.DlrWarn(f"diff max: {diff_max}")

        ok = 0; error = 0
        if status:
           ok += 1
           _dlr.DlrInfo(f"OK {out_data.shape}")
           if verbose:
               _dlr.DlrInfo(f"in_data\n{in_data}")
               _dlr.DlrInfo(f"in_kernel\n{in_kernel}")
               _dlr.DlrInfo(f"out_data\n{out_data}")
               _dlr.DlrInfo(f"nout_data\n{nout_data}")
        else:
           error += 1
           _dlr.DlrError(f"Mis-match {out_data.shape}")
           if verbose:
               _dlr.DlrInfo(f"in_data\n{in_data}")
               _dlr.DlrInfo(f"in_kernel\n{in_kernel}")
               _dlr.DlrInfo(f"out_data\n{out_data}")
               _dlr.DlrInfo(f"nout_data\n{nout_data}")

        return True if error==0 else False

        return False

    def GenDataDeconv2d(data, kernel, bias, rigor=False, verbose=False):
        if (data.dim()==4): # minibatch x channel x size x size
            in_minibatch = data.shape[0]
            in_channel   = data.shape[1]
            in_size      = data.shape[2]
        else: return None
        out_channel = kernel.shape[1]
        kerne_size  = kernel.shape[2]
        bias_size   = bias.shape[0]
        error  = 0
        if (in_channel!=kernel.shape[0]): error += 1
        if (bias is not None) and (out_channel!=bias_size): error += 1

        in_data   = (100+100)*torch.rand(size=data.shape) - 100
        in_kernel = ( 10+ 10)*torch.rand(size=kernel.shape) - 10
        if (bias is not None): in_bias = 5+5*torch.rand(size=bias.shape) - 5
        else: in_bias = None

        return in_data, in_kernel, in_bias

#===============================================================================
if __name__=='__main__':
    def TestNormBatch(dtype=torch.float32
                     ,limit=1.E-3 # error limit
                     ,random=False
                     ,rigor=False
                     ,verbose=False):
        dim = 1
        if dim==1: # to test 1D batch_norm
            minibatches = 3
            channels    = 2
            sizes       = 4
            ndims       = [minibatches, channels, sizes]
            raw_data, raw_mean, raw_var, raw_std = GenDataNorm(ndims, plot=False, rigor=rigor, verbose=verbose)
        elif dim==2: # to test 2D batch_norm
            minibatches = 2
            channels    = 2
            rows        = 4
            cols        = 3
            ndims       = [minibatches, channels, rows, cols]
            raw_data, raw_mean, raw_var, raw_std = GenDataNorm(ndims, plot=False, rigor=rigor, verbose=verbose)
        elif dim==3: # to test 3D batch_norm
            # not fully tested
            minibatches = 1
            channels    = 1
            depths      = 1
            rows        = 4
            cols        = 3
            ndims       = [minibatches, channels, depths, rows, cols]
            raw_data, raw_mean, raw_var, raw_std = GenDataNorm(ndims, plot=False, rigor=rigor, verbose=verbose)
        #std_data = (raw_data - raw_mean)/raw_std

        out_data  = F.batch_norm( input=raw_data
                                , running_mean=raw_mean
                                , running_var=raw_var
                                , weight=None
                                , bias=None
                                , training=False
                                , momentum=1.0
                                , eps=1E-5)
        nout_data = batch_norm ( input=raw_data
                               , running_mean=raw_mean
                               , running_var=raw_var
                               , weight=None
                               , bias=None
                               , eps=1E-5
                               , rigor=rigor
                               , verbose=verbose)
        status = False
        if (out_data is not None) and (nout_data is not None):
            status = torch.all(torch.lt(torch.abs(torch.add(out_data, -nout_data)), limit))
            if not status:
                diff_max = torch.max(torch.abs(torch.add(out_data, -nout_data)))
                _dlr.DlrWarn(f"diff max: {diff_max}")

        ok = 0; error = 0
        if status:
           ok += 1
           _dlr.DlrInfo(f"OK {out_data.shape}")
           if verbose:
               _dlr.DlrInfo(f"raw_data\n{raw_data}")
               _dlr.DlrInfo(f"out_data\n{out_data}")
               _dlr.DlrInfo(f"nout_data\n{nout_data}")
        else:
           error += 1
           _dlr.DlrError(f"Mis-match {out_data.shape}")
           if verbose:
               _dlr.DlrInfo(f"raw_data\n{raw_data}")
               _dlr.DlrInfo(f"out_data\n{out_data}")
               _dlr.DlrInfo(f"nout_data\n{nout_data}")

        return True if error==0 else False

    #---------------------------------------------------------------------------
    def GenDataNorm(ndims, plot=False, rigor=False, verbose=False):
        import numpy as np
        import matplotlib.pyplot as plt
        if (len(ndims)==3): # minibatch x channel x size
            minibatches = ndims[0]
            channels    = ndims[1]
            sizes       = ndims[2]
        elif (len(ndims)==4): # minibatch x channel x size x size
            minibatches = ndims[0]
            channels    = ndims[1]
            sizes       = ndims[2]
        elif (len(ndims)==5): # minibatch x channel x depth x size x size
            minibatches = ndims[0]
            channels    = ndims[1]
            depth       = ndims[2]
            sizes       = ndims[3]
        raw_data = (100+100)*torch.rand(size=ndims)-100

        raw_mean= torch.zeros(size=[channels]) # mean value
        raw_var = torch.zeros(size=[channels]) # variance
        raw_std = torch.zeros(size=[channels]) # standard-deviation
        #mean/var/std should be one for each channel regardless minibanch
        if len(ndims)==5: # should take care of depth
            # not fully tested
            raw_mean = torch.mean(input=raw_data[0], axis=(2,-1)) # mean value
            raw_var  = torch.var (input=raw_data[0], axis=(2,-1)) # variance
            raw_std  = torch.std (input=raw_data[0], axis=(2,-1)) # standard-deviation=sqrt(var)
        else: # minibatch x channel x size
            raw_mean = torch.mean(input=raw_data[0], axis=(1,-1)) # mean value
            raw_var  = torch.var (input=raw_data[0], axis=(1,-1)) # variance
            raw_std  = torch.std (input=raw_data[0], axis=(1,-1)) # standard-deviation=sqrt(var)
        #std_data = (raw_data - raw_mean)/raw_std
        #_dlr.DlrInfo(f"std_data ={std_data}")

        if plot:
            if True:
                plt.subplot(1, 2, 1)
                plt.hist(raw_data, bins=50)

                plt.subplot(1, 2, 2)
                plt.hist(std_data, bins=50)
                plt.show()
            else:
                bins = 50
                raw_hist, raw_bin = np.histogram(raw_data, bins=bins)

                std_hist, std_bin = np.histogram(std_data, bins=bins)

                num = int(raw_data.numel()/2)
                x = torch.linspace(start=-num, end=num, steps=num*2)
                y = torch.flatten(raw_data)
                plt.subplot(3, 1, 1)
                plt.plot(x, y)

                a = np.linspace(min(raw_bin), max(raw_bin), bins)
                b = raw_hist
                plt.subplot(3, 1, 2)
                plt.plot(a, b)

                n = np.linspace(min(std_bin), max(std_bin), bins)
                m = std_hist
                plt.subplot(3, 1, 3)
                plt.plot(n, m)

                plt.show()

        return raw_data, raw_mean, raw_var, raw_std


#===============================================================================
if __name__=='__main__':
    def TestActivations(func='ReLu' # DLR function name
                       ,tfunc='relu' # PyTorch Functional function name
                       ,negative_slope=0.01
                       ,dtype=torch.int32
                       ,limit=1E-3 # error limit
                       ,random=False
                       ,rigor=False
                       ,verbose=False):
        """
        dtype: specify data type of data one of {torch.int32, torch.float32, torch.float64}
        """
        func_name = tfunc
        if func=='Sigmoid' or func=='Tanh': function  = getattr(torch, tfunc)
        else: function  = getattr(F, tfunc)

        if random:
            minibatch = (torch.randint(low=1, high=3, size=[1], dtype=torch.int)).data.numpy()
            d         = (torch.randint(low=1, high=10, size=[1], dtype=torch.int)).data.numpy()
            dims      = (torch.randint(low=1, high=10, size=tuple(d), dtype=torch.int)).data.numpy()
        else:
            minibatch = [1, 2, 3]
            dims      = [1, 2, 3] # 1=1-dimension, 2=2-dimension

        ok = 0; error = 0
        for dim in dims:
            ndim = (torch.randint(low=1, high=10, size=[dim], dtype=torch.int)).data.numpy() # [x] or [x, y] or [x, y, z]
            in_data = (100+100)*torch.rand(size=tuple(ndim))-100
            if dtype is torch.int32:
                in_data = in_data.type(torch.int32)

            if func == 'LeakyReLu': # dealing with "not implemented for 'Int'"
                if dtype is torch.int32:
                   in_data = in_data.type(torch.float32)
                out_data = function(input=in_data, negative_slope=negative_slope)
                nout_data = globals()[func_name]( in_data
                                                , rigor=rigor
                                                , verbose=verbose)
                if dtype is torch.int32:
                   in_data = in_data.type(torch.int32)
                   nout_data = nout_data.type(torch.int32)
            else:
                if (func!='ReLu') and (dtype==torch.int32):
                   in_data = in_data.type(torch.float32)
                out_data = function(in_data)
                nout_data = globals()[func_name]( in_data
                                                , rigor=rigor
                                                , verbose=verbose)
                if (func!='ReLu') and (dtype==torch.int32):
                   in_data = in_data.type(torch.int32)
                   out_data = out_data.type(torch.int32)
                   nout_data = nout_data.type(torch.int32)

            if dtype is torch.int32:
                out_data  = out_data.type(torch.int32)
                nout_data = nout_data.type(torch.int32)

            
            status = False
            if (out_data is not None) and (nout_data is not None):
                status = torch.all(torch.lt(torch.abs(torch.add(out_data, -nout_data)), limit))
                if not status:
                    diff_max = torch.max(torch.abs(torch.add(out_data, -nout_data)))
                    _dlr.DlrWarn(f"diff max: {diff_max}")

            if status:
               ok += 1
               _dlr.DlrError(f"OK {out_data.shape}")
               if verbose:
                   _dlr.DlrInfo(f"in_data\n{in_data}")
                   _dlr.DlrInfo(f"nout_data\n{nout_data}")
                   _dlr.DlrInfo(f"out_data\n{out_data}")
            else:
               error += 1
               _dlr.DlrError(f"Mis-match {out_data.shape}")
               if verbose:
                   _dlr.DlrInfo(f"in_data\n{in_data}")
                   _dlr.DlrInfo(f"nout_data\n{nout_data}")
                   _dlr.DlrInfo(f"out_data\n{out_data}")
        return True if error==0 else False

    def TestActivationReLu(dtype, random, limit, rigor, verbose):
        return TestActivations('ReLu', 'relu', dtype=dtype, limit=limit, random=random, rigor=rigor, verbose=verbose)
    def TestActivationLeakyReLu(negative_slope, dtype, limit, random, rigor, verbose):
        return TestActivations('LeakyReLu', 'leaky_relu', negative_slope=negative_slope, dtype=dtype, limit=limit, random=random, rigor=rigor, verbose=verbose)
    def TestActivationTanh(dtype, random, limit, rigor, verbose):
        return TestActivations('Tanh', 'tanh', dtype=dtype, limit=limit, random=random, rigor=rigor, verbose=verbose)
    def TestActivationSigmoid(dtype, random, limit, rigor, verbose):
        return TestActivations('Sigmoid', 'sigmoid', dtype=dtype, limit=limit, random=random, rigor=rigor, verbose=verbose)

#===============================================================================

if __name__=='__main__':
    import sys
    if 'torch' not in sys.modules:
        _dlr.DlrError("PyTorch is not loaded.")

    import argparse
    parser = argparse.ArgumentParser(description='DLR PyTorch Testing')

    parser.add_argument('--layer', dest='layer', type=str, default='ReLu',
                        help='Specify layer to test (default: ReLu)\n'
                            +'ReLu LeakyReLu Tanh Sigmoid\n'
                            +'Convolution2d Pooling2dMax Pooling2dAvg\n'
                            +'Linear1d Linear2d Concat2d\n'
                            +'NormBatch'+'Deconvlution2d'
                       )
    parser.add_argument('--limit', dest='limit', type=float, default=1.0E-3,
                        help='Specify error limmit (default: 1.0E-3)')
    parser.add_argument('--nslope', dest='negative_slope', type=float, default=0.01,
                        help='Specify negative slope of LeakyReLU (default: 0.01)')
    parser.add_argument('--dtype', dest='dtype', type=str, default='int32',
                        help='Specify data type (default: int32) float32, float64')
    parser.add_argument('--random', dest='random', action='store_true', default=False,
                        help='Use random pattern (default: False)')
    parser.add_argument('--rigor', dest='rigor', action='store_true', default=False,
                        help='Check values rigorously (default: False)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='Verbose (default: False)')

    args = parser.parse_args()
    random = args.random
    rigor = args.rigor
    verbose = args.verbose
    limit   = args.limit
    negative_slope = args.negative_slope
    dtype = { 'int32'   : torch.int32,
              'float32' : torch.float32,
              'float64' : torch.float64
            } [args.dtype]
    layer = args.layer
    func  = { 'Convolution2d'  : TestConvolution2d      
            , 'Pooling2dMax'   : TestPooling2dMax       
            , 'Pooling2dAvg'   : TestPooling2dAvg       
            , 'Linear1d'       : TestLinear1d           
            , 'LinearNd'       : TestLinearNd           
            , 'Concat2d'       : TestConcat2d           
            , 'ReLu'           : TestActivationReLu     
            , 'LeakyReLu'      : TestActivationLeakyReLu
            , 'Tanh'           : TestActivationTanh     
            , 'Sigmoid'        : TestActivationSigmoid  
            , 'NormBatch'      : TestNormBatch         
            , 'Deconvolution2d': TestDeconvolution2d
            } [layer]

    _dlr.DlrPrint("Testing " + layer, flush=True)
    if layer == 'LeakyReLu':
       status = func(negative_slope=negative_slope,dtype=dtype,limit=limit,random=random,rigor=rigor,verbose=verbose)
    else:
       status = func(dtype=dtype,limit=limit,random=random,rigor=rigor,verbose=verbose)
    #func_name = layer
    #status = locals()[func_name](dtype=dtype,random=random,rigor=rigor,verbose=verbose)

#===============================================================================
# Revision history:
#
# 2020.09.30: Started by Ando Ki (adki@future-ds.com)
#===============================================================================
