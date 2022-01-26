
#include "concat_2d.hpp"

extern "C" {

void Concat2dInt
(           int      *out_data // depends on dim; size of (in_rowsA*in_colsA+in_rowsB*in_colsB)
    , const int      *in_dataA // in_rowsA x in_colsA
    , const int      *in_dataB // in_rowsB x in_colsB
    , const uint16_t  in_rowsA // height
    , const uint16_t  in_colsA // width
    , const uint16_t  in_rowsB // height
    , const uint16_t  in_colsB // width
    , const uint8_t   dim      // 0 or 1
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
)
{
    dlr::Concat2d<int>
    (     out_data
        , in_dataA
        , in_dataB
        , in_rowsA
        , in_colsA
        , in_rowsB
        , in_colsB
        , dim
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void Concat2dFloat
(           float    *out_data // depends on dim; size of (in_rowsA*in_colsA+in_rowsB*in_colsB)
    , const float    *in_dataA // in_rowsA x in_colsA
    , const float    *in_dataB // in_rowsB x in_colsB
    , const uint16_t  in_rowsA // height
    , const uint16_t  in_colsA // width
    , const uint16_t  in_rowsB // height
    , const uint16_t  in_colsB // width
    , const uint8_t   dim      // 0 or 1
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
)
{
    dlr::Concat2d<float>
    (     out_data
        , in_dataA
        , in_dataB
        , in_rowsA
        , in_colsA
        , in_rowsB
        , in_colsB
        , dim
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

void Concat2dDouble
(           double   *out_data // depends on dim; size of (in_rowsA*in_colsA+in_rowsB*in_colsB)
    , const double   *in_dataA // in_rowsA x in_colsA
    , const double   *in_dataB // in_rowsB x in_colsB
    , const uint16_t  in_rowsA // height
    , const uint16_t  in_colsA // width
    , const uint16_t  in_rowsB // height
    , const uint16_t  in_colsB // width
    , const uint8_t   dim      // 0 or 1
    #if !defined(__SYNTHESIS__)
    , const int       rigor       // check rigorously when 1
    , const int       verbose     // verbose level
    #endif
)
{
    dlr::Concat2d<double>
    (     out_data
        , in_dataA
        , in_dataB
        , in_rowsA
        , in_colsA
        , in_rowsB
        , in_colsB
        , dim
        #if !defined(__SYNTHESIS__)
        , rigor
        , verbose
        #endif
    );
}

} // extern "C"
