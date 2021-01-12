
#include "norm_1d_batch.hpp"

extern "C" {

void Norm1dBatchInt
(           int      *out_data // in_channel x in_size (contiguous)
    , const int      *in_data  // in_channel x in_size (contiguous)
    , const int      *running_mean // in_channel (contiguous)
    , const int      *running_var  // in_channel (contiguous)
    , const int      *scale // NULL or in_channel (contiguous)
    , const int      *bias // NULL or in_channel (contiguous)
    , const uint32_t  in_size // num of elements per channel
    , const uint16_t  scale_size // 0 or in_channel
    , const uint16_t  bias_size // 0 or in_channel
    , const uint16_t  in_channel // 1 or n
    , const float     epsilon // default: 1E-5
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
)
{
    Norm1dBatch<int>
    (     out_data
        , in_data
        , running_mean
        , running_var
        , scale
        , bias
        , in_size
        , scale_size
        , bias_size
        , in_channel
        , epsilon
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void Norm1dBatchFloat
(           float    *out_data // in_channel x in_size (contiguous)
    , const float    *in_data  // in_channel x in_size (contiguous)
    , const float    *running_mean // in_channel (contiguous)
    , const float    *running_var  // in_channel (contiguous)
    , const float    *scale // NULL or in_channel (contiguous)
    , const float    *bias // NULL or in_channel (contiguous)
    , const uint32_t  in_size // num of elements per channel
    , const uint16_t  scale_size // 0 or in_channel
    , const uint16_t  bias_size // 0 or in_channel
    , const uint16_t  in_channel // 1 or n
    , const float     epsilon // default: 1E-5
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
)
{
    Norm1dBatch<float>
    (     out_data
        , in_data
        , running_mean
        , running_var
        , scale
        , bias
        , in_size
        , scale_size
        , bias_size
        , in_channel
        , epsilon
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void Norm1dBatchDouble
(           double   *out_data // in_channel x in_size (contiguous)
    , const double   *in_data  // in_channel x in_size (contiguous)
    , const double   *running_mean // in_channel (contiguous)
    , const double   *running_var  // in_channel (contiguous)
    , const double   *scale // NULL or in_channel (contiguous)
    , const double   *bias // NULL or in_channel (contiguous)
    , const uint32_t  in_size // num of elements per channel
    , const uint16_t  scale_size // 0 or in_channel
    , const uint16_t  bias_size // 0 or in_channel
    , const uint16_t  in_channel // 1 or n
    , const float     epsilon // default: 1E-5
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
)
{
    Norm1dBatch<double>
    (     out_data
        , in_data
        , running_mean
        , running_var
        , scale
        , bias
        , in_size
        , scale_size
        , bias_size
        , in_channel
        , epsilon
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

} // extern "C"
