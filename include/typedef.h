#pragma once

using ImageSize_t = std::pair<long, long>;

typedef struct {
    long channel;
    long width;
    long height;
} ImageDims_t;

typedef struct {
    long l;
    long t;
    long r;
    long b;
} ImagePadding_t;