
#include "activation_sigmoid.hpp"

extern "C" {

void ActivationSigmoidInt
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
    dlr::ActivationSigmoid<int>
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

void ActivationSigmoidFloat
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
    dlr::ActivationSigmoid<float>
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

void ActivationSigmoidDouble
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
    dlr::ActivationSigmoid<double>
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
