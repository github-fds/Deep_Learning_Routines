#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


#define Concat2d  Concat2dFloat

extern void Concat2dInt
(           int      *out_data // depend on dim
    , const int      *in_dataA // in_rowsA x in_colsA
    , const int      *in_dataB // in_rowsB x in_colsB
    , const uint16_t  in_rowsA // height
    , const uint16_t  in_colsA // width
    , const uint16_t  in_rowsB // height
    , const uint16_t  in_colsB // width
    , const uint8_t   dim      // 0 or 1
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Concat2dFloat
(           float    *out_data // depend on dim
    , const float    *in_dataA // in_rowsA x in_colsA
    , const float    *in_dataB // in_rowsB x in_colsB
    , const uint16_t  in_rowsA // height
    , const uint16_t  in_colsA // width
    , const uint16_t  in_rowsB // height
    , const uint16_t  in_colsB // width
    , const uint8_t   dim      // 0 or 1
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Concat2dDouble
(           double   *out_data // depend on dim
    , const double   *in_dataA // in_rowsA x in_colsA
    , const double   *in_dataB // in_rowsB x in_colsB
    , const uint16_t  in_rowsA // height
    , const uint16_t  in_colsA // width
    , const uint16_t  in_rowsB // height
    , const uint16_t  in_colsB // width
    , const uint8_t   dim      // 0 or 1
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

#ifdef __cplusplus
}
#endif
