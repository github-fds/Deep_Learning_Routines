#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file activation_relu.hpp
 * @brief This file contains non-linear activation routine.
 * @author FDS
 * @date Oct. 20, 2020
 */
#include <stdint.h>
#if !defined(__SYNTHESIS__)
#include <stdio.h>
#include <assert.h>
#include <typeinfo>
#include "dlr_common.h"
#endif

template<class TYPE=float>
void ActivationReLu
(           TYPE     *out_data // contiguous: channel x size
    , const TYPE     *in_data  // contiguous: channel x size
    , const uint32_t  size     // number of elements per channel
    , const uint16_t  channel
    #if !defined(__SYNTHESIS__)
    , const int       rigor=0    // check rigorously when 1
    , const int       verbose=0  // verbose level
    #endif
)
{
    #if !defined(__SYNTHESIS__)
    #define QuoteIdent(ident) #ident
    #define QuoteMacro(macro) QuoteIdent(macro)
    if (verbose) {
      //dlrInfo("function      =%s\n", QuoteMacro(ACTIVATION_FUNCTION));
        dlrInfo("data type     =%s\n", QuoteMacro(TYPE));
        dlrInfo("size          =%d\n", size);
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert (size>0);
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #else
    #pragma HLS inline off
    #endif

    uint16_t c;
    uint32_t s;

#if 1
    for (c=0; c<channel; ++c){
        for (s=0; s<size; ++s){
            TYPE *pX = (TYPE*)(in_data +c*size+s);
            TYPE *pZ = (TYPE*)(out_data+c*size+s);
            TYPE value = *pX;
            *pZ = (value<=(TYPE)0) ? (TYPE)0 : value;
        } // for (s=0;
    } // for (c=0;
#else
    TYPE *pX = (TYPE*)in_data ;
    TYPE *pZ = (TYPE*)out_data;
    for (c=0; c<channel; ++c){
        for (s=0; s<size; ++s){
            TYPE value = *pX;
            *pZ = (value<=(TYPE)0) ? (TYPE)0 : value;
            ++pX;
            ++pZ;
        } // for (s=0;
    } // for (c=0;
#endif
}

// Z[n]=f(X[n])

/*
 * Revision history
 *
 * 2020.10.20: 'channel' added
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 */
