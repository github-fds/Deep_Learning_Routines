
#include "activation_leakyrelu.hpp"

extern "C" {

void ActivationLeakyReLuInt
(           int      *out_data // channel x size x size
    , const int      *in_data  // channel x size x size
    , const uint32_t  size
    , const uint16_t  channel
    , const uint32_t  negative_slope
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
)
{
    dlr::ActivationLeakyReLu<int>
    (     out_data
        , in_data
        , size
        , channel
        , negative_slope
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void ActivationLeakyReLuFloat
(           float    *out_data
    , const float    *in_data
    , const uint32_t  size
    , const uint16_t  channel
    , const uint32_t  negative_slope
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
)
{
    dlr::ActivationLeakyReLu<float>
    (     out_data
        , in_data
        , size
        , channel
        , negative_slope
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void ActivationLeakyReLuDouble
(           double   *out_data
    , const double   *in_data
    , const uint32_t  size
    , const uint16_t  channel
    , const uint32_t  negative_slope
    #if !defined(__SYNTHESIS__)
    , const int       rigor
    , const int       verbose
    #endif
)
{
    dlr::ActivationLeakyReLu<double>
    (     out_data
        , in_data
        , size
        , channel
        , negative_slope
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

} // extern "C"
