#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ActivationLeakyReLu ActivationLeakyReLuFloat

extern void ActivationLeakyReLuInt
(           int      *out_data
    , const int      *in_data
    , const uint32_t  size
    , const uint16_t  channel
    , const uint32_t  negative_slope
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
);

extern void ActivationLeakyReLuFloat
(           float    *out_data
    , const float    *in_data
    , const uint32_t  size
    , const uint16_t  channel
    , const uint32_t  negative_slope
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
);

extern void ActivationLeakyReLuDouble
(           double   *out_data
    , const double   *in_data
    , const uint32_t  size
    , const uint16_t  channel
    , const uint32_t  negative_slope
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
);

#ifdef __cplusplus
}
#endif
