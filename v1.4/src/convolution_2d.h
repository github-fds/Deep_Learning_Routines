#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Convolution2d Convolution2dFloat

extern void Convolution2dInt
(           int      *out_data    // out_channel x out_size x out_size
    , const int      *in_data     // in_channel x in_size x in_size
    , const int      *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const int      *bias        // bias per kernel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // number of biases, it should be the same as out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride      // stride default 1
    , const uint8_t   padding     // padding default 0
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Convolution2dFloat
(           float    *out_data    // out_channel x out_size x out_size
    , const float    *in_data     // in_channel x in_size x in_size
    , const float    *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const float    *bias        // bias per kernel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // number of biases, it should be the same as out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride      // stride default 1
    , const uint8_t   padding     // padding default 0
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Convolution2dDouble
(           double   *out_data    // out_channel x out_size x out_size
    , const double   *in_data     // in_channel x in_size x in_size
    , const double   *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const double   *bias        // bias per kernel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // number of biases, it should be the same as out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride      // stride default 1
    , const uint8_t   padding     // padding default 0
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

#ifdef __cplusplus
}
#endif
