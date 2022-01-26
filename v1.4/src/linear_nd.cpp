
#include "linear_nd.hpp"

extern "C" {

void LinearNdInt
(           int      *out_data // ndim x out_size
    , const int      *in_data  // ndim x in_size
    , const int      *weight   // out_size x in_size
    , const int      *bias     // out_size
    , const uint16_t  out_size // num of elements per dim
    , const uint16_t  in_size
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    dlr::LinearNd<int, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             , ndim
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void LinearNdFloat
(           float    *out_data // ndim x out_size
    , const float    *in_data  // ndim x in_size
    , const float    *weight   // out_size x in_size
    , const float    *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    dlr::LinearNd<float, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             , ndim
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void LinearNdDouble
(           double   *out_data // ndim x out_size
    , const double   *in_data  // ndim x in_size
    , const double   *weight   // out_size x in_size
    , const double   *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    dlr::LinearNd<double, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             , ndim
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void LinearNdIntReLu
(           int      *out_data // ndim x out_size
    , const int      *in_data  // ndim x in_size
    , const int      *weight   // out_size x in_size
    , const int      *bias     // out_size
    , const uint16_t  out_size // num of elements per dim
    , const uint16_t  in_size
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    dlr::LinearNd<int, 1> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             , ndim
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void LinearNdFloatReLu
(           float    *out_data // ndim x out_size
    , const float    *in_data  // ndim x in_size
    , const float    *weight   // out_size x in_size
    , const float    *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    dlr::LinearNd<float, 1> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             , ndim
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void LinearNdDoubleReLu
(           double   *out_data // ndim x out_size
    , const double   *in_data  // ndim x in_size
    , const double   *weight   // out_size x in_size
    , const double   *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    , const uint8_t   ndim
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    dlr::LinearNd<double, 1> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             , ndim
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

} // extern "C"
