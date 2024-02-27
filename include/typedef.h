#pragma once

using ImageSize_t = std::pair<long, long>;

typedef struct
{
    long channel;
    long width;
    long height;
} ImageDims_t;

typedef struct
{
    long l;
    long t;
    long r;
    long b;
} ImagePadding_t;

enum class torch_dtype
{
    float32,
    float64,
    int32,
    int64,
    uint8,
    int8,
    int16,
    _float,
    _int
};