#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Deconvolution2d Deconvolution2dFloat

extern void Deconvolution2dInt
(           int      *out_data    // out_channel x out_size x out_size
    , const int      *in_data     // in_channel x in_size x in_size
    , const int      *kernel      // out_channel x kernel_size x kernel_size
    , const int      *bias
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint8_t   bias_size
    , const uint16_t  in_channel
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding
    #if !defined(__SYNTHESIS__)
    , const int       rigor   // check rigorously when 1
    , const int       verbose // verbose level
    #endif
);

extern void Deconvolution2dFloat
(           float    *out_data    // out_channel x out_size x out_size
    , const float    *in_data     // in_channel x in_size x in_size
    , const float    *kernel      // out_channel x kernel_size x kernel_size
    , const float    *bias
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint8_t   bias_size
    , const uint16_t  in_channel
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding
    #if !defined(__SYNTHESIS__)
    , const int       rigor   // check rigorously when 1
    , const int       verbose // verbose level
    #endif
);

extern void Deconvolution2dDouble
(           double   *out_data    // out_channel x out_size x out_size
    , const double   *in_data     // in_channel x in_size x in_size
    , const double   *kernel      // out_channel x kernel_size x kernel_size
    , const double   *bias
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint8_t   bias_size
    , const uint16_t  in_channel
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding
    #if !defined(__SYNTHESIS__)
    , const int       rigor   // check rigorously when 1
    , const int       verbose // verbose level
    #endif
);

#ifdef __cplusplus
}
#endif
