
#include "linear_1d.hpp"

extern "C" {

void Linear1dInt
(           int      *out_data // out_size
    , const int      *in_data  // in_size
    , const int      *weight   // out_size x in_size
    , const int      *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    Linear1d<int, 0, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void Linear1dFloat
(           float    *out_data // out_size
    , const float    *in_data  // in_size
    , const float    *weight   // out_size x in_size
    , const float    *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    Linear1d<float, 0, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void Linear1dDouble
(           double   *out_data // out_size
    , const double   *in_data  // in_size
    , const double   *weight   // out_size x in_size
    , const double   *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    Linear1d<double, 0, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void Linear1dIntReLu
(           int      *out_data // out_size
    , const int      *in_data  // in_size
    , const int      *weight   // out_size x in_size
    , const int      *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    Linear1d<int, 1, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void Linear1dFloatReLu
(           float    *out_data // out_size
    , const float    *in_data  // in_size
    , const float    *weight   // out_size x in_size
    , const float    *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    Linear1d<float, 1, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

void Linear1dDoubleReLu
(           double   *out_data // out_size
    , const double   *in_data  // in_size
    , const double   *weight   // out_size x in_size
    , const double   *bias     // out_size
    , const uint16_t  out_size
    , const uint16_t  in_size
    , const uint16_t  bias_size
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
    )
{
    Linear1d<double, 1, 0> ( out_data
             , in_data
             , weight
             , bias
             , out_size
             , in_size
             , bias_size
             #if !defined(__SYNTHESIS__)
             , rigor
             , verbose 
             #endif
             );
}

} // extern "C"
