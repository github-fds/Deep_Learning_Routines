#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Linearnd LinearndFloat

extern void LinearNdInt
(           int      *out_data    // out_feature
    , const int      *in_data     // in_feature
    , const int      *weight      // out_feature x in_feature
    , const int      *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void LinearNdFloat
(           float    *out_data    // out_feature
    , const float    *in_data     // in_feature
    , const float    *weight      // out_feature x in_feature
    , const float    *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void LinearNdDouble
(           double   *out_data    // out_feature x 1
    , const double   *in_data     // in_feature x 1
    , const double   *weight      // out_feature x in_feature
    , const double   *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

#ifdef __cplusplus
}
#endif
