#pragma once

#include <cstddef>

template <typename T>
constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
