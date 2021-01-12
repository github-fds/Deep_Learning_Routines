#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file linear_1d.hpp
 * @brief This file contains 1 dimensional linear (fully-connected) routine.
 * @author FDS
 * @date Aug. 31, 2020
 */
#include <stdint.h>
#if !defined(__SYNTHESIS__)
#include <stdio.h>
#include <assert.h>
#include <typeinfo>
#include "dlr_common.h"
#endif

template< class TYPE=float
        , int ReLu=0
        , const int LeakyReLu=0
        , const uint32_t negative_slope=0x3DCCCCCD // 0.01
        >
void Linear1d
(           TYPE    *out_data // out_size
    , const TYPE    *in_data  // in_size
    , const TYPE    *weight   // out_size x in_size
    , const TYPE    *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
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
      //dlrInfo("data type  =%s\n",    QuoteMacro(TYPE));
      //dlrInfo("data type  =%s\n", typeid(TYPE).name());
        dlrInfo("out_size   =%d\n",    out_size );
        dlrInfo("in_size    =%d\n",    in_size  );
        dlrInfo("weight_size=%dx%d\n", out_size, in_size);
        dlrInfo("bias_size  =%d\n",    bias_size   );
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert (in_size>0);
        assert (out_size>0);
        assert ((bias_size==0)||(out_size==bias_size));
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #endif

    uint16_t o, i;

    for(o=0; o<out_size; ++o){
        TYPE  sum = (TYPE)0;
        TYPE *pZ = (TYPE*)(out_data+o);
        TYPE B = (bias_size==(TYPE)0) ? (TYPE)0 : *(bias+o);
        for(i=0; i<in_size; ++i){
            TYPE *pX = (TYPE*)(in_data+i);
            TYPE *pW = (TYPE*)(weight +(o*in_size + i));
            sum += (*pX)*(*pW);
        }
        if (ReLu) {
            *pZ = ((sum+B)<=(TYPE)0) ? (TYPE)0 : (sum+B);
        } else if (LeakyReLu) {
             uint32_t ss = negative_slope;
             float slope = *((float *)&ss); // make sure that it is 32-bit wide item
            *pZ = (sum<(TYPE )0) ? ((float)sum*slope) : sum;
        } else {
            *pZ = sum+B;
        }
    }
}

// Z=out_data[out_size]
// X=in_data[in_size]
// W=weight[out_size][in_size]
// B=bias[out_size]
// Z=X*W'+B, where W' is transposed W.

/*
 * Revision history
 *
 * 2020.11.12: 'LeakyReLu' template added.
 * 2020.09.20: parameter order of bias and bias_size changed.
 *             parameter 'rigor' and 'verbose' added.
 * 2020.08.31: Updated by participants of 2020 Summer Intern Program.
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 */
