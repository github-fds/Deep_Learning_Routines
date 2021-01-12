
#include "activation_relu.hpp"

extern "C" {

void ActivationReLuInt
(           int      *out_data
    , const int      *in_data
    , const uint32_t  size
    , const uint16_t  channel
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
)
{
    ActivationReLu<int>
    (     out_data
        , in_data
        , size
        , channel
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void ActivationReLuFloat
(           float    *out_data
    , const float    *in_data
    , const uint32_t  size
    , const uint16_t  channel
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
)
{
    ActivationReLu<float>
    (     out_data
        , in_data
        , size
        , channel
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void ActivationReLuDouble
(           double   *out_data
    , const double   *in_data
    , const uint32_t  size
    , const uint16_t  channel
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
)
{
    ActivationReLu<double>
    (     out_data
        , in_data
        , size
        , channel
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

} // extern "C"
