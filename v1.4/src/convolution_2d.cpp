
#include "convolution_2d.hpp"

extern "C" {

void Convolution2dInt
(           int      *out_data    // out_channel x out_size x out_size
    , const int      *in_data     // in_channel x in_size x in_size
    , const int      *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const int      *bias        // out_channel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding
    #if !defined(__SYNTHESIS__)
    , const int       rigor   // check rigorously when 1
    , const int       verbose // verbose level
    #endif
)
{
    dlr::Convolution2d<int> ( out_data
                  , in_data
                  , kernel
                  , bias
                  , out_size
                  , in_size
                  , kernel_size
                  , bias_size
                  , in_channel
                  , out_channel
                  , stride
                  , padding
                  #if !defined(__SYNTHESIS__)
                  , rigor
                  , verbose
                  #endif
                  );
}

void Convolution2dFloat
(           float    *out_data    // out_channel x out_size x out_size
    , const float    *in_data     // in_channel x in_size x in_size
    , const float    *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const float    *bias        // out_channel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding
    #if !defined(__SYNTHESIS__)
    , const int       rigor   // check rigorously when 1
    , const int       verbose // verbose level
    #endif
)
{
    dlr::Convolution2d<float> ( out_data
                  , in_data
                  , kernel
                  , bias
                  , out_size
                  , in_size
                  , kernel_size
                  , bias_size
                  , in_channel
                  , out_channel
                  , stride
                  , padding
                  #if !defined(__SYNTHESIS__)
                  , rigor
                  , verbose
                  #endif
                  );
}

void Convolution2dDouble
(           double   *out_data    // out_channel x out_size x out_size
    , const double   *in_data     // in_channel x in_size x in_size
    , const double   *kernel      // in_channel x out_channel x kernel_size x kernel_size
    , const double   *bias        // out_channel
    , const uint16_t  out_size    // only for square matrix
    , const uint16_t  in_size     // only for square matrix
    , const uint8_t   kernel_size // only for square matrix
    , const uint16_t  bias_size   // out_channel
    , const uint16_t  in_channel  // number of input channels
    , const uint16_t  out_channel // number of filters (kernels)
    , const uint8_t   stride
    , const uint8_t   padding
    #if !defined(__SYNTHESIS__)
    , const int       rigor   // check rigorously when 1
    , const int       verbose // verbose level
    #endif
)
{
    dlr::Convolution2d<double> ( out_data
                  , in_data
                  , kernel
                  , bias
                  , out_size
                  , in_size
                  , kernel_size
                  , bias_size
                  , in_channel
                  , out_channel
                  , stride
                  , padding
                  #if !defined(__SYNTHESIS__)
                  , rigor
                  , verbose
                  #endif
                  );
}

} // extern "C"
