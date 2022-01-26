
#include "pooling_2d_max.hpp"

extern "C" {

void Pooling2dMaxInt
(           int      *out_data    // out_channel x out_size x out_size
    , const int      *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel     // in/out channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode   // not implemented yet
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
    )
{
    Pooling2dMax<int, 0, 0>( out_data
                , in_data
                , out_size
                , in_size
                , kernel_size
                , channel
                , stride
                , padding
                , ceil_mode
                #if !defined(__SYNTHESIS__)
                , rigor
                , verbose
                #endif
                );
}

void Pooling2dMaxFloat
(           float    *out_data    // out_channel x out_size x out_size
    , const float    *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel     // in/out channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode   // not implemented yet
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
    )
{
    Pooling2dMax<float, 0, 0>( out_data
                , in_data
                , out_size
                , in_size
                , kernel_size
                , channel
                , stride
                , padding
                , ceil_mode
                #if !defined(__SYNTHESIS__)
                , rigor
                , verbose
                #endif
                );
}

void Pooling2dMaxDouble
(           double   *out_data    // out_channel x out_size x out_size
    , const double   *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel     // in/out channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode   // not implemented yet
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
    )
{
    Pooling2dMax<double, 0, 0>( out_data
                , in_data
                , out_size
                , in_size
                , kernel_size
                , channel
                , stride
                , padding
                , ceil_mode
                #if !defined(__SYNTHESIS__)
                , rigor
                , verbose
                #endif
                );
}

void Pooling2dMaxIntReLu
(           int      *out_data    // out_channel x out_size x out_size
    , const int      *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel     // in/out channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode   // not implemented yet
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
    )
{
    Pooling2dMax<int, 1, 0>( out_data
                , in_data
                , out_size
                , in_size
                , kernel_size
                , channel
                , stride
                , padding
                , ceil_mode
                #if !defined(__SYNTHESIS__)
                , rigor
                , verbose
                #endif
                );
}

void Pooling2dMaxFloatReLu
(           float    *out_data    // out_channel x out_size x out_size
    , const float    *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel     // in/out channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode   // not implemented yet
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
    )
{
    Pooling2dMax<float, 1, 0>( out_data
                , in_data
                , out_size
                , in_size
                , kernel_size
                , channel
                , stride
                , padding
                , ceil_mode
                #if !defined(__SYNTHESIS__)
                , rigor
                , verbose
                #endif
                );
}

void Pooling2dMaxDoubleReLu
(           double   *out_data    // out_channel x out_size x out_size
    , const double   *in_data     // in_channel x in_size x in_size
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  channel     // in/out channel
    , const uint8_t   stride
    , const uint8_t   padding
    , const int       ceil_mode   // not implemented yet
    #if !defined(__SYNTHESIS__)
    , const int       rigor // check rigorously when 1
    , const int       verbose
    #endif
    )
{
    Pooling2dMax<double, 1, 0>( out_data
                , in_data
                , out_size
                , in_size
                , kernel_size
                , channel
                , stride
                , padding
                , ceil_mode
                #if !defined(__SYNTHESIS__)
                , rigor
                , verbose
                #endif
                );
}

} // extern "C"
