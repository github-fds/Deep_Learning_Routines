
#include "activation_tanh.hpp"

extern "C" {

void ActivationTanhInt
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
    ActivationTanh<int>
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

void ActivationTanhFloat
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
    ActivationTanh<float>
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

void ActivationTanhDouble
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
    ActivationTanh<double>
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
