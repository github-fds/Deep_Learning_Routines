#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Pooling2dAvg Pooling2dAvgFloat

extern void Pooling2dAvgInt
(           int      *out_data    // out_channel x out_size x out_size
    , const int      *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
);

extern void Pooling2dAvgFloat
(           float    *out_data    // out_channel x out_size x out_size   
    , const float    *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
);

extern void Pooling2dAvgDouble
(           double   *out_data    // out_channel x out_size x out_size
    , const double   *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
);

#ifdef __cplusplus
}
#endif
