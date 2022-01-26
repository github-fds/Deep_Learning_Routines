#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file linear_nd.hpp
 * @brief This file contains n dimensional linear (fully-connecte) routine.
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

namespace dlr { // deep learning routines

template<class TYPE=float, int ReLu=0>
void LinearNd
(           TYPE    *out_data // ndim x out_size
    , const TYPE    *in_data  // ndim x in_size
    , const TYPE    *weight   // out_size x in_size
    , const TYPE    *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    , const uint8_t   ndim // in_data and out_data is ndim
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
      //dlrInfo("data type  =%s\n",    typeid(TYPE).name());
        dlrInfo("in_ndim    =%s\n",    ndim);
        dlrInfo("out_size   =%d\n",    out_size );
        dlrInfo("in_size    =%d\n",    in_size  );
        dlrInfo("weight_size=%dx%d\n", out_size, in_size);
        dlrInfo("bias_size  =%d\n",    bias_size   );
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert (ndim>0);
        assert (in_size>0);
        assert (out_size>0);
        assert ((bias_size==0)||(out_size==bias_size));
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #endif

    uint8_t n;
    uint16_t o, i;

    for (n=0; n<ndim; ++n) {
        for(o=0; o<out_size; ++o){
            TYPE  sum = (TYPE)0;
            TYPE *pZ = (TYPE*)(out_data+n*out_size+o);
            TYPE B = (bias_size==(TYPE)0) ? (TYPE)0 : *(bias+o);
            for(i=0; i<in_size; ++i){
                TYPE *pX = (TYPE*)(in_data+n*in_size+i);
                TYPE *pW = (TYPE*)(weight +(o*in_size
                                            + i));
                sum += (*pX)*(*pW);
            } // for (i=0
            if (ReLu) *pZ = ((sum+B)<=(TYPE)0) ? (TYPE)0 : (sum+B);
            else      *pZ = sum+B;
        } // for (o=0;
    } // for (n=0;
}

// Z=out_data[ndim][out_size]
// X=in_data[ndim][in_size]
// W=weight[out_size][in_size]
// B=bias[out_size]
// Z[n]=X[n]*W'+B, where W' is transposed W.

} // namespace dlr
/*
 * Revision history
 *
 * 2020.09.20: parameter order of bias and bias_size changed.
 *             parameter 'rigor' and 'verbose' added.
 * 2020.08.31: Updated by participants of 2020 Summer Intern Program.
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 */
