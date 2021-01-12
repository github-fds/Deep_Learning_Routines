#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file activation_leakyrelu.hpp
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

template<class TYPE =float>
void ActivationLeakyReLu
(           TYPE     *out_data // contiguous: channel x size x size
    , const TYPE     *in_data  // contiguous: channel x size x size
    , const uint32_t  size     // number of elements per channel
    , const uint16_t  channel  // number of channels
    , const uint32_t  negative_slope=0x3DCCCCCD // negative slope for LeakyReLu
                               // 0.01 by default (0x)(0011111110000100011110101110000101000111101011100001010001111011)
                               // make sure it is 32-bit bit-pattern,
                               // which can be float32 or int32 or fixed32 ...
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
      //dlrInfo("data type  =%s\n", typeid(TYPE ).name());
        dlrInfo("data type     =%s\n", QuoteMacro(TYPE ));
        dlrInfo("size          =%u\n", size);
        dlrInfo("channel       =%u\n", channel);
        dlrInfo("negative slope=%f\n", *(float*)&negative_slope);
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert (size>0);
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #endif

    uint16_t c;
    uint32_t s;

    TYPE  *pX = (TYPE *)in_data;
    TYPE  *pZ = (TYPE *)out_data;
    float slope = *((float *)&negative_slope); // make sure that it is 32-bit wide item
    for (c=0; c<channel; ++c){
        for (s=0; s<size; ++s){
            TYPE  value = *pX;
           *pZ = (value<(TYPE )0) ? ((float)value*slope) : value;
            pX++;
            pZ++;
        } // for (s=0;
    } // for (c=0;
}

// Z[n]=f(X[n])

/*
 * Revision history
 *
 * 2020.10.20: 'channel' added.
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 */
