#pragma once
/*
 * Copyright (c) 2019-2020-2021 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file dlr_common.h
 * @brief This file contains common routines to deal with message.
 * @author FDS
 * @date Oct. 4, 2021
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
 * 2021.10.04: basename() used.
 * 2020.09.20: Started by Ando Ki (adki@future-ds.com)
 */
