#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file dlr_common.h
 * @brief This file contains 2 dimensional convolution routine.
 * @author FDS
 * @date Oct. 01, 2020
 */
#ifdef __cplusplus
extern "C" {
#endif

#define __FFL__   __FILE__,__LINE__,__func__

#define  dlrError(...)  dlrErrorCore(__FFL__, ##__VA_ARGS__)
#define  dlrWarn(...)   dlrWarnCore (__FFL__, ##__VA_ARGS__)
#define  dlrInfo(...)   dlrInfoCore (__FFL__, ##__VA_ARGS__)
#define  dlrPrint(...)  dlrPrintCore(__FFL__, ##__VA_ARGS__)

extern void dlrErrorCore(const char *filename, const int lnum, const char *funcname, const char *fmt, ...);
extern int  dlrWarnCore (const char *filename, const int lnum, const char *funcname, const char *fmt, ...);
extern int  dlrInfoCore (const char *filename, const int lnum, const char *funcname, const char *fmt, ...);
extern int  dlrPrintCore(const char *filename, const int lnum, const char *funcname, const char *fmt, ...);

#ifdef __cplusplus
}
#endif
/*
 * Revision history
 *
 * 2020.09.20: Started by Ando Ki (adki@future-ds.com)
 */
