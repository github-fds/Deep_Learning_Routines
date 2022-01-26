#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Linear1d Linear1dFloat

extern void Linear1dInt
(           int      *out_data    // out_feature
    , const int      *in_data     // in_feature
    , const int      *weight      // out_feature x in_feature
    , const int      *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Linear1dFloat
(           float    *out_data    // out_feature
    , const float    *in_data     // in_feature
    , const float    *weight      // out_feature x in_feature
    , const float    *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Linear1dDouble
(           double   *out_data    // out_feature x 1
    , const double   *in_data     // in_feature x 1
    , const double   *weight      // out_feature x in_feature
    , const double   *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

#define Linear1dReLu Linear1dFloatReLu

extern void Linear1dIntReLu
(           int      *out_data    // out_feature
    , const int      *in_data     // in_feature
    , const int      *weight      // out_feature x in_feature
    , const int      *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Linear1dFloatReLu
(           float    *out_data    // out_feature
    , const float    *in_data     // in_feature
    , const float    *weight      // out_feature x in_feature
    , const float    *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

extern void Linear1dDoubleReLu
(           double   *out_data    // out_feature x 1
    , const double   *in_data     // in_feature x 1
    , const double   *weight      // out_feature x in_feature
    , const double   *bias
    , const uint16_t  out_size
    , const uint16_t  in_size 
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
);

#ifdef __cplusplus
}
#endif
