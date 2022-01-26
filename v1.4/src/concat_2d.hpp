#pragma once
/*
 * Copyright (c) 2019-2020 by Future Design Systems.
 * All right reserved.
 * http://www.future-ds.com
 *
 * @file concat_2d.c
 * @brief This file contains 2 dimensional concatenation.
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
void Concat2d
(           TYPE    *out_data // depends on dim; size of (in_rowsA*in_colsA+in_rowsB*in_colsB)
    , const TYPE    *in_dataA // in_rowsA x in_colsA
    , const TYPE    *in_dataB // in_rowsB x in_colsB
    , const uint16_t  in_rowsA // height
    , const uint16_t  in_colsA // width
    , const uint16_t  in_rowsB // height
    , const uint16_t  in_colsB // width
    , const uint8_t   dim      // 0 or 1
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
      //dlrInfo("data type  =%s\n", QuoteMacro(TYPE));
      //dlrInfo("data type  =%s\n", typeid(TYPE).name());
        dlrInfo("out_size   =%d\n", in_rowsA*in_colsA+in_rowsB*in_colsB);
        dlrInfo("in_rowsA   =%d\n", in_rowsA );
        dlrInfo("in_colsA   =%d\n", in_colsA );
        dlrInfo("in_rowsB   =%d\n", in_rowsB );
        dlrInfo("in_colsB   =%d\n", in_colsB );
        dlrInfo("dim        =%d\n", dim);
        fflush(stderr); fflush(stdout);
    }
    if (rigor) {
        assert ((dim>=0)&&(dim<2));
        assert ((in_rowsA>0)&&(in_colsA>0));
        assert ((in_rowsB>0)&&(in_colsB>0));
        if (dim==0) assert(in_colsA==in_colsB); // width should be the same
        if (dim==1) assert(in_rowsA==in_rowsB); // height should be the same
    }
    #undef QuoteMacro
    #undef QuoteIdent
    #endif

    if (dim==0) { // concate two array back to back
        //  Concat([in_rowsA;in_colsA],[inb_rows;inB_colw],0)
        //         ==> out_data[in_rowsA+in_rowsB;in_colsA]
        //         where in_colsA and in_colsB should be the same
        // 
        //  |<--- in_colsA --->|
        //  +------------------+ -
        //  |                  | |
        //  +------------------+ |            |<--- in_colsA --->|
        //  |                  | in_rowsA     +------------------+ -
        //  +------------------+ |            |                  | |
        //  |                  | |            +------------------+ |
        //  +------------------+ -            |                  | in_rowsA
        //                             ====>  +------------------+ |
        //  |<--- in_colsB --->|              |                  | |
        //  +------------------+ -            +------------------+ -
        //  |                  | |            |                  | |
        //  +------------------+ |            +------------------+ |
        //  |                  | in_rowsB     |                  | in_rowsB
        //  +------------------+ |            +------------------+ |
        //  |                  | |            |                  | |
        //  +------------------+ |            +------------------+ |
        //  |                  | |            |                  | |
        //  +------------------+ -            +------------------+ -

        #if 0
            void *pt = (void *)out_data;
            uint32_t nbytes = in_rowsA*in_colsA*sizeof(TYPE);
            memcpy(pt, (void*)in_dataA, (size_t)nbytes);
            pt += nbytes;
            nbytes = in_rowsB*in_colsB*sizeof(TYPE);
            memcpy((void*)pt, (void*)in_dataB, (size_t)nbytes);
        #else
        uint16_t r, c;
        TYPE *pt = (TYPE *)out_data;
        for (r=0; r<in_rowsA; ++r) {
            for (c=0; c<in_colsA; ++c) {
                *pt = *(TYPE*)in_dataA;
                 pt++;
                 in_dataA++;
            }
        }
        for (r=0; r<in_rowsB; ++r) {
            for (c=0; c<in_colsB; ++c) {
                *pt = *(TYPE*)in_dataB;
                 pt++;
                 in_dataB++;
            }
        }
        #endif
    } else { // dim==1
        //  Concat([in_rowsA;in_colsA],[inb_rows;inB_colw],1)
        //         ==> out_data[in_rowsA;in_colsA+in_colsB]
        //         where in_rowsA and in_rowsB should be the same
        // 
        //  |<--- in_colsA --->|
        //  +------------------+ -
        //  |                  | |
        //  +------------------+ |                |<--- in_colsA --->|<--- in_colsB ---------->|
        //  |                  | in_rowsA         +------------------+-------------------------+ -
        //  +------------------+ |                |                  |                         | |
        //  |                  | |                +------------------+-------------------------+ |
        //  +------------------+ -                |                  |                         | in_rowsA
        //                                 ====>  +------------------+-------------------------+ |
        //  |<--- in_colsB ---------->|           |                  |                         | |
        //  +-------------------------+ -         +------------------+-------------------------+ _
        //  |                         | |
        //  +-------------------------+ |
        //  |                         | in_rowsB (==in_rowsA)
        //  +-------------------------+ |          
        //  |                         | |          
        //  +-------------------------+ -          

        #if 0
            uint16_t r;
            void *pt  = (void *)out_data;
            void *ptA = (void *)in_dataA;
            void *ptB = (void *)in_dataB;
            uint32_t nbytesA=in_colsA*sizeof(TYPE);
            uint32_t nbytesB=in_colsB*sizeof(TYPE);
            for (r=0; r<in_rowsA; ++r) {
                 memcpy(pt, ptA, (size_t)nbytesA);
                 pt  += nbytesA;
                 ptA += nbytesA;
                 memcpy(pt, ptB, (size_t)nbytesB);
                 pt  += nbytesB;
                 ptB += nbytesB;
            }
        #else
        uint16_t r, c;
        TYPE *pt = (TYPE *)out_data;
        for (r=0; r<in_rowsA; ++r) {
            for (c=0; c<in_colsA; ++c) {
                *pt = *(TYPE*)in_dataA;
                 pt++;
                 in_dataA++;
            }
            for (c=0; c<in_colsB; ++c) {
                *pt = *(TYPE*)in_dataB;
                 pt++;
                 in_dataB++;
            }
        }
        #endif
    }
}

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
