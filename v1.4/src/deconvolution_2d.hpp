#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file deconvolution_2d.hpp
 * @brief This file contains 2 dimensional deconvolution, i.e. transpose convolution
 * @author FDS
 * @date Aug. 31, 2020
 */
#include <stdint.h>
#include <string.h>
#if !defined(__SYNTHESIS__)
#include <stdio.h>
#include <assert.h>
#include <typeinfo>
#include "dlr_common.h"
#endif

namespace dlr { // deep learning routines

template<class TYPE=float>
void Deconvolution2d
(           TYPE     *out_data    // out_channel x out_size x out_size
    , const TYPE     *in_data     // in_channel x in_size x in_size
    , const TYPE     *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const TYPE     *bias
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint8_t   bias_size
    , const uint16_t  in_channel
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding=0
    #if !defined(__SYNTHESIS__)
    , const int       rigor=0 // check rigorously when 1
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
        dlrInfo("out_size      =%u:%u\n", out_size, ((in_size-1)*stride-2*padding+(kernel_size-1)+1));
        dlrInfo("in_size       =%u\n", in_size          );
        dlrInfo("kernel_size   =%u\n", kernel_size      );
        dlrInfo("in_channel    =%u\n", in_channel       );
        dlrInfo("out_channel   =%u\n", out_channel      );
        dlrInfo("stride        =%u\n", stride           );
        dlrInfo("padding       =%u\n", padding          );
        dlrInfo("bias_size     =%u\n", bias_size        );
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert (in_channel>0);
        assert (out_channel>0);
        assert (out_size==((in_size-1)*stride-2*padding+(kernel_size-1)+1));
        //assert (out_size==((in_size-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1));
        //assert ((kernel_size%2)==1);
        assert (out_size>0);
        assert (stride>0);
        //assert (padding<=(kernel_size/2));
        //assert (output_padding < dilation && output_padding < stride);
        assert ((bias_size==0)||(out_channel==bias_size));
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #endif

    #define UpperPadding(CURSOR, PADDING)          (CURSOR <   PADDING)
    #define LowerPadding(CURSOR, PADDING, IN_SIZE) (CURSOR >= (PADDING+IN_SIZE))
    #define LeftPadding( CURSOR, PADDING)          (CURSOR <   PADDING)
    #define RightPadding(CURSOR, PADDING, IN_SIZE) (CURSOR >= (PADDING+IN_SIZE))
    #define IsPadding(W, H, P, IS) ( UpperPadding(H, P)\
                                   ||LowerPadding(H, P, IS)\
                                   ||LeftPadding( W, P)\
                                   ||RightPadding(W, P, IS) )

    uint16_t ch, f;
    uint16_t i, j, k, g, r, c;
    uint16_t t_width, t_height;
    uint16_t t_padded_size = out_size + 2*padding;;

    for (f=0; f<out_channel; ++f) {
        TYPE B = (bias_size==(TYPE)0) ? (TYPE)0 : *(bias+f);
        TYPE *pZ = (TYPE*)(out_data+(f*out_size*out_size));
        for (r=0; r<out_size; ++r) {
            for (c=0; c<out_size; ++c) {
                #if 1
                *pZ = B;
                 pZ++;
                #else
                *pZ++ = B; // not working
                *pZ = B; pZ += 1; // not working
                *pZ = B; pZ++; // not working
                 pZ[r*out_size+c] = B;
                #endif
            }
        }

        TYPE *pX;
        TYPE *pW;
        for (ch=0; ch<in_channel; ++ch) {
            for (i=0, r=0; i<in_size; ++i, r+=stride){
                for (j=0, c=0; j<in_size; ++j, c+=stride){
                    pX = (TYPE*)(in_data+(ch*in_size*in_size+i*in_size+j));
                    for (g=0; g<kernel_size; ++g){
                        for (k=0; k<kernel_size; ++k){
                            TYPE accum = (TYPE)0;
                            t_width = j+c;
                            t_height = r+i;
                            pZ = (TYPE*)(out_data+(f*out_size*out_size)
                                                  +((r+g-padding)*out_size
                                                  +c+k-padding));
                            if (padding==0 || !IsPadding(t_width, t_height, padding, out_size)) {
                                 pW = (TYPE*)(  kernel+(ch*out_channel*kernel_size*kernel_size)
                                                       +(f*kernel_size*kernel_size)
                                                       +(g*kernel_size
                                                       +k));
                                 accum += (*pX)*(*pW);
                            }
                            *pZ += accum;
                        }
                    }
                }
            }
        }
    }

    #undef UpperPadding
    #undef LowerPadding
    #undef LeftPadding
    #undef RightPadding
    #undef IsPadding
}

// output padding is assymmetric. it’s only applied in the right and the bottom of the image.

} // namespace dlr
/*
 * Revision history
 *
 * 2020.11.12: '*pZ++ = B' HLS pointer arithmetic bug-fixed
 * 2020.09.20: parameter order of bias and bias_size changed.
 *             parameter 'rigor' and 'verbose' added.
 * 2020.08.31: Updated by participants of 2020 Summer Intern Program.
 *             - ChaeEon Lim; GeunSu Song; YoonSeong Lim;
 * 2020.07.01: Started by Ando Ki (adki@future-ds.com)
 */
