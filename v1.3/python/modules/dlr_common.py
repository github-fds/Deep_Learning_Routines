#!/usr/bin/env python
"""
This file contains common part of Python interface of DLR for PyTorch.
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
__revision__   = "0"
__maintainer__ = "Ando Ki"
__email__      = "contact@future-ds.com"
__status__     = "Development"
__date__       = "2020.04.25"
__description__= "Python interface of DLR for PyTorch"

#-------------------------------------------------------------------------------
import os
import sys
import traceback
import inspect
import ctypes
import ctypes.util

#===============================================================================
# utility functions
# print GetFunctionName()+'('+str(GetFunctionParametersAndValues())+')'
def GetLineno():
    """
    Returns the current line number in the program.
    :return: integer of the line number.
    """
    return inspect.currentframe().f_back.f_lineno

def GetFunctionName():
    """
    Returns the current function name in the program.
    :return: string of the function name.
    """
    return traceback.extract_stack(None, 2)[0][2]

def GetFunctionParametersAndValues():
    """
    Returns the dictionary of function arguments in the program.
    :return: disctionary of function arguments.
    """
    frame = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame)
    return ([(i, values[i]) for i in args])

def DlrError(*args, **kwargs):
    print(traceback.extract_stack(None, 2)[0][2], "Error: ".join(map(str,args)), **kwargs)

def DlrWarn(*args, **kwargs):
    print(traceback.extract_stack(None, 2)[0][2], "Warning: ".join(map(str,args)), **kwargs)

def DlrInfo(*args, **kwargs):
    print(traceback.extract_stack(None, 2)[0][2], "Info: ".join(map(str,args)), **kwargs)

def DlrPrint(*args, **kwargs):
    print(traceback.extract_stack(None, 2)[0][2], " ".join(map(str,args)), **kwargs)

#-------------------------------------------------------------------------------
# utility functions: function signature or function wrapper
def WrapFunction(lib, funcname, restype, argtypes):
    """
    Simplify wrapping ctypes functions
    :param lib: library (object returned from ctypes.CDLL()
    :param funcname: string of function name
    :param restype: type of return value
    :param argtypes: a list of types of the function arguments
    :return: Python object holding function, restype and argtypes.
    """
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

#===============================================================================
#verbose = True
#verboseprint = print if verbose else lambda *a, **k: None

#===============================================================================
# let check 'DLR_HOME' environment variable
if 'DLR_HOME' not in os.environ:
   print (GetFunctionName(), "Warning: the environment variable DLR_HOME not defined.", flush=True)
   DLR_HOME='.'
else:
   DLR_HOME=os.environ["DLR_HOME"]

_libdlr = os.path.abspath( os.path.join(DLR_HOME, "lib/libdlr.so"))

#-------------------------------------------------------------------------------
if not os.path.isfile(_libdlr):
   print (GetFunctionName(), _libdlr+' is not a file', flush=True)
   traceback.print_exc(file=sys.stdout)
   sys.exit(1)
else:
   # '__debug__' This constant is true if Python was not started with an -O option
   if __debug__: print (GetFunctionName(), _libdlr+" found.", flush=True)

#-------------------------------------------------------------------------------
try:
    _dlr = ctypes.CDLL(_libdlr)
except:
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)

#-------------------------------------------------------------------------------
# need debug for this 'rigor' and 'verbose'
rigor = False
verbose = False

def set_rigor( ri ): rigor = ri
def get_rigor(): return rigor

def set_verbose ( ve ): verbose = ve
def get_verbose (): return verbose

#===============================================================================
# Revision history:
#
# 2020.04.58: Started        by Ando Ki     (adki@future-ds.com)
#===============================================================================
