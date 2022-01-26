#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file norm_1d_batch.hpp
 * @brief This file contains batch normalization
 * @author FDS
 * @date Aug. 31, 2020
 */
#include <stdint.h>
#include <math.h>
#if !defined(__SYNTHESIS__)
#include <stdio.h>
#include <assert.h>
#include <typeinfo>
#include "dlr_common.h"
#endif

template<class TYPE=float>
void Norm1dBatch
(           TYPE     *out_data // in_channel x in_size (contiguous)
    , const TYPE     *in_data  // in_channel x in_size (contiguous)
    , const TYPE     *running_mean // in_channel (contiguous) [mean]
    , const TYPE     *running_var  // in_channel (contiguous) [variance, not deviation]
    , const TYPE     *scale // NULL or in_channel (contiguous) [gamma: scaling factor]
    , const TYPE     *bias // NULL or in_channel (contiguous) [beta: shift factor]
    , const uint32_t  in_size // num of elements per channel
    , const uint16_t  scale_size // 0 or in_channel
    , const uint16_t  bias_size // 0 or in_channel
    , const uint16_t  in_channel // 1 or n
    , const float     epsilon=1E-5 // default: 1E-5 (noise for reguarization)
    #if !defined(__SYNTHESIS__)
    , const int       rigor=0     // check rigorously when 1
    , const int       verbose=0   // verbose level
    #endif
)
{
    #if !defined(__SYNTHESIS__)
    #define QuoteIdent(ident) #ident
    #define QuoteMacro(macro) QuoteIdent(macro)
    if (verbose) {
        dlrInfo("data type  =%s\n", QuoteMacro(TYPE));
        dlrInfo("in_channel =%u\n", in_channel       );
        dlrInfo("in_size    =%u\n", in_size          );
        dlrInfo("scale_size =%u\n", scale_size       );
        dlrInfo("bias_size  =%u\n", bias_size        );
        dlrInfo("epsilon    =%f\n", epsilon          );
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert (in_channel>0);
        assert (in_size>0);
        assert ((scale_size==0)||(scale_size==in_channel));
        assert ((bias_size==0)||(bias_size==in_channel));
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #endif

    uint16_t  f;
    uint32_t  s;

    TYPE *pX = (TYPE*)(in_data);
    TYPE *pZ = (TYPE*)(out_data);
    for (f=0; f<in_channel; ++f) {
        TYPE B     = (bias_size==0) ? (TYPE)0 : *(TYPE *)(bias+f);
        TYPE mean  = *(running_mean+f);
        TYPE var   = *(running_var+f);
        TYPE S     = (scale_size==0) ? (TYPE)1 : *(TYPE *)(scale+f);
        for (s=0; s<in_size; ++s) {
             *pZ = (TYPE)((((*pX)-mean) / (sqrt(var+epsilon))) * S + B);
             ++pX;
             ++pZ;
        } 
    }
}

/*
 * Revision history
 *
 * 2020.09.20: parameter order of bias and bias_size changed.
 *             parameter 'rigor' and 'verbose' added.
 * 2020.08.31: Updated by participants of 2020 Summer Intern Program.
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 */
