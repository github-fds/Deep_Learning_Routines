/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file dlr_common.cpp
 * @brief This file contains 2 dimensional convolution routine.
 * @author FDS
 * @date Aug. 31, 2020
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "dlr_common.h"

#define stream_error stderr
#define stream_warn  stdout
#define stream_info  stdout
#define stream_std   stdout

void dlrErrorCore(const char *file,
              const int line,
              const char *func,
              const char *fmt, ...) {
    va_list     ap;
    int vfprintf(FILE* stream, const char *fmt, va_list ap);
    fflush(stream_info); fflush(stream_warn);
    fprintf(stream_error, "Error: %s %d %s(): ", file, line, func);
    va_start(ap, fmt);
    vfprintf(stream_error, fmt, ap);
    va_end(ap);
    exit(1);
}

int dlrWarnCore(const char *file,
            const int line,
            const char *func,
            const char *fmt, ...) {
    va_list     ap;
    int         ret;
    int vfprintf(FILE* stream, const char *fmt, va_list ap);
    fprintf(stream_warn, "Warning: %s %d %s(): ", file, line, func);
    va_start(ap, fmt);
    ret = vfprintf(stream_warn, fmt, ap);
    va_end(ap);
    return(ret);
}

int dlrInfoCore(const char *file,
            const int line,
            const char *func,
            const char *fmt, ...) {
    va_list     ap;
    int         ret;
    int vfprintf(FILE* stream, const char *fmt, va_list ap);
    fprintf(stream_info, "Info: %s %d %s(): ", file, line, func);
    va_start(ap, fmt);
    ret = vfprintf(stream_info, fmt, ap);
    va_end(ap);
    return(ret);
}

int dlrPrintCore(const char *file,
            const int line,
            const char *func,
            const char *fmt, ...) {
    va_list ap;
    int     ret;
    int vfprintf(FILE* stream, const char *fmt, va_list ap);
    fprintf(stream_std, "%s %d %s(): ", file, line, func);
    va_start(ap, fmt);
    ret = vfprintf(stream_info, fmt, ap);
    va_end(ap);
    return(ret);
}

/*
 * Revision history
 *
 * 2020.09.20: Started by Ando Ki (adki@future-ds.com)
 */
