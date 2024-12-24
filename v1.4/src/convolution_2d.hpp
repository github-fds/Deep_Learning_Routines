#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file convolution_2d.hpp
 * @brief This file contains 2 dimensional convolution routine.
 * @author FDS
 * @date Oct. 23, 2020
 */
#include <stdint.h>
#if !defined(__SYNTHESIS__)
#include <stdio.h>
#include <assert.h>
#include <typeinfo>
#include "dlr_common.h"
#endif

namespace dlr { // deep learning routines

template<class TYPE=float>
void Convolution2d
(           TYPE     *out_data    // out_channel x out_size x out_size
    , const TYPE     *in_data     // in_channel x in_size x in_size
    , const TYPE     *kernel      // out_channel x in_channel x kernel_size x kernel_size
    , const TYPE     *bias        // out_channel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding=0
    #if !defined(__SYNTHESIS__)
    , const int       rigor=0   // check rigorously when 1
    , const int       verbose=0 // verbose level
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
        dlrInfo("bias_size  =%d\n", bias_size   );
        dlrInfo("in_channel =%d\n", in_channel  );
        dlrInfo("out_channel=%d\n", out_channel );
        dlrInfo("stride     =%d\n", stride      );
        dlrInfo("padding    =%d\n", padding     );
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert (in_channel>0);
        assert (out_channel>0);
        assert (out_size==(((in_size-kernel_size+2*padding)/stride)+1));
        assert ((kernel_size%2)==1);
        assert (stride>0);
        assert (padding>=0);
        assert (padding<=(kernel_size/2));
        assert ((bias_size==0)||(out_channel==bias_size));
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

    uint16_t ch, f;
    uint16_t i, j, k, g, r, c;
    uint16_t t_width, t_height;
    const uint16_t t_padded_size = in_size + 2*padding;
    const uint16_t in_height=in_size;
    const uint16_t in_width=in_size;
    const uint16_t out_height=out_size;
    const uint16_t out_width=out_size;
    const uint16_t kernel_height=kernel_size;
    const uint16_t kernel_width=kernel_size;

    //#pragma GCC unroll f
    //#pragma GCC ivdep
    for (f=0; f<out_channel; ++f) {
        TYPE B = (bias_size==(TYPE)0) ? (TYPE)0 : *(bias+f);
        TYPE *pZ = (TYPE*)(out_data+(f*out_height*out_width));
        for (r=0; r<out_height; ++r) {
            for (c=0; c<out_width; ++c) {
                #if 1
                *pZ = B;
                 pZ++; // working for LeNet-5
                #else
                *pZ++ = B; // not working
                *pZ = B; pZ += 1; // not working
                *pZ = B; pZ++; // not working
                 pZ[r*out_height+c] = B; // working ro Tiny-Yolo-V2
                #endif
            } // for (c=0;
        } // for (r=0;
        for (ch=0; ch<in_channel; ++ch) {
            for (g=0, r=0; r<(t_padded_size - kernel_height + 1); ++g, r += stride) {
                for (k=0, c=0; c<(t_padded_size - kernel_width + 1); ++k, c += stride) {
                    TYPE accum=(TYPE)0;
                    pZ = (TYPE*)(out_data+((f*out_height*out_width)+(g*out_width+k)));
                    for (i=0; i<kernel_height; ++i) {
                        for (j=0; j<kernel_width; ++j) {
                            t_width = j+c;
                            t_height = r+i;
                            if (padding==0 || !IsPadding(t_width, t_height, padding, in_height, in_width)) {
                                 TYPE *pX = (TYPE*)(in_data+(ch*in_height*in_width
                                                             +(r+i-padding)*in_width
                                                             +j+c-padding));
                                 TYPE *pW = (TYPE*)( kernel+(f*in_channel*kernel_height*kernel_width
                                                             +ch*kernel_height*kernel_width
                                                             +i*kernel_width
                                                             +j));
                                accum += (*pX)*(*pW);
                            } // if (padding==0
                        } // for (j=0;
                    } // for (i=0
                    *pZ += accum;
                } // for (k=0
            } // for (g=0;
        } // for (ch=0;
    } // for (f=0;

    #undef UpperPadding
    #undef LowerPadding
    #undef LeftPadding
    #undef RightPadding
    #undef IsPadding
}

//  out_data[out_channel][out_size][out_size]
//  in_data[in_channel][in_size][in_size]
//  kernel[out_channel][in_channel][kernel_size][kernel_size] # note the order of dimensions
//  bias[out_channel]

} // namespace dlr
/*
 * Revision history
 *
 * 2020.11.12: '*pZ++ = B' HLS pointer arithmetic bug-fixed
 * 2020.10.23: 't_current' size bug-fixed by using 't_width' and 't_height'.
 * 2020.10.01: C++ template version by Ando Ki.
 * 2020.09.20: parameter order of bias and bias_size changed.
 *             parameter 'rigor' and 'verbose' added.
 * 2020.08.31: Updated by participants of 2020 Summer Intern Program.
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 */
