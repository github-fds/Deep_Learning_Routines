#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file pooling_2d_avg.hpp
 * @brief This file contains 2 dimensional average pooling routine.
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
void Pooling2dAvg
(           TYPE    *out_data    // out_channel x out_size x out_size
    , const TYPE    *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel     // in/out channel
    , const uint8_t   stride
    , const uint8_t   padding=0
    , const int       ceil_mode=0   // not implemented yet
    #if !defined(__SYNTHESIS__)
    , const int       rigor=0 // check rigorously when 1
    , const int       verbose=0
    #endif
    )
{
    #if !defined(__SYNTHESIS__)
    #define QuoteIdent(ident) #ident
    #define QuoteMacro(macro) QuoteIdent(macro)
    if (verbose) {
      //dlrInfo("data type  =%s\n", QuoteMacro(TYPE));
      //dlrInfo("data type  =%s\n", typeid(TYPE).name());
        dlrInfo("out_size   =%d\n", out_size    );
        dlrInfo("in_size    =%d\n", in_size     );
        dlrInfo("kernel_size=%d\n", kernel_size );
        dlrInfo("channel    =%d\n", channel     );
        dlrInfo("stride     =%d\n", stride      );
        dlrInfo("padding    =%d\n", padding     );
        dlrInfo("ceil       =%d\n", ceil_mode   );
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        uint16_t expect=(ceil_mode) ? (int)ceil(((in_size-kernel_size+(2*padding))/stride)+1+in_size%kernel_size)
                                    : (int)floor(((in_size-kernel_size+(2*padding))/stride)+1);
        if (out_size!=expect) dlrWarn("out_size mis-match: %u, but %u expected\n", out_size, expect);
        assert ((kernel_size%2)==0);
        assert (stride>0);
        assert (padding>=0);
        assert (padding<=(kernel_size/2));
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #endif

    #define UpperPadding(CURSOR, PADDING)          (CURSOR <   PADDING)
    #define LowerPadding(CURSOR, PADDING, IN_SIZE) (CURSOR >= (PADDING+IN_SIZE))
    #define LeftPadding( CURSOR, PADDING)          (CURSOR <   PADDING)
    #define RightPadding(CURSOR, PADDING, IN_SIZE) (CURSOR >= (PADDING+IN_SIZE))
    #define IsPadding(W, H, P, IH, IW) ( UpperPadding(H, P)\
                                       ||LowerPadding(H, P, IH)\
                                       ||LeftPadding( W, P)\
                                       ||RightPadding(W, P, IW) )
    uint16_t ch;
    uint16_t i, j, k, g, r, c;
    uint16_t t_width, t_height;
    const uint16_t t_padded_size = in_size + 2*padding;
    const uint16_t in_width=in_size; // num of columns
    const uint16_t in_height=in_size; // num of rows
    const uint16_t out_width=out_size;
    const uint16_t out_height=out_size;
    const uint16_t kernel_width=kernel_size;
    const uint16_t kernel_height=kernel_size;

    TYPE *pZ = out_data;
    for (ch=0; ch<channel; ++ch) {
        for (g=0, r=0; g<out_height; ++g, r+=stride) {
            for (k=0, c=0; k<out_width; ++k, c+=stride) {
                TYPE avg=(TYPE)0;
                for (i=0; i<kernel_height; ++i) {
                     TYPE *pX = (TYPE*)(in_data+(ch*in_height*in_width
                                                 +(r+i-padding)*in_width
                                                 +c-padding));
                    for (j=0; j<kernel_width; ++j) {
                        t_width = j+c;
                        t_height = r+i;
                        if (padding==0 || !IsPadding(t_width, t_height, padding, in_height, in_width)) {
                            TYPE *pX = (TYPE*)(in_data+(ch*in_height*in_width
                                                        +(r+i-padding)*in_width
                                                        +j+c-padding));
                            avg += *pX;
                        }
                        ++pX;
                    }
                }
                *pZ = avg/(kernel_height*kernel_width);
                ++pZ;
            }
        }
    }
    #undef UpperPadding
    #undef LowerPadding
    #undef LeftPadding
    #undef RightPadding
    #undef IsPadding
}

/*
 * Revision history
 *
 * 2020.09.20: parameter order of bias and bias_size changed.
 * 2020.08.31: Updated by participants of 2020 Summer Intern Program.
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 */
