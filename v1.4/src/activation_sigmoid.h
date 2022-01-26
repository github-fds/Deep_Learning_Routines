#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ActivationSigmoid ActivationSigmoidFloat

extern void ActivationSigmoidInt
(           int      *out_data
    , const int      *in_data
    , const uint32_t  size
    , const uint16_t  channel
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
);

extern void ActivationSigmoidFloat
(           float    *out_data
    , const float    *in_data
    , const uint32_t  size
    , const uint16_t  channel
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
);

extern void ActivationSigmoidDouble
(           double   *out_data
    , const double   *in_data
    , const uint32_t  size
    , const uint16_t  channel
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
);

#ifdef __cplusplus
}
#endif
