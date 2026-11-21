#include <float.h>
#include <stdint.h>
#include <string.h>
#include <gurabo/tensor.h>
#include "float16_bits.h"

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
               FLT_MAX_EXP == 128, "Gurabo requires IEEE binary32 float");

static uint32_t f32_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float f32_from_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t gurabo_f16_from_f32(float value)
{
    return gb_f16_from_f32_bits(f32_bits(value));
}

float gurabo_f16_to_f32(uint16_t bits)
{
    return f32_from_bits(gb_f32_bits_from_f16(bits));
}

uint16_t gurabo_bf16_from_f32(float value)
{
    return gb_bf16_from_f32_bits(f32_bits(value));
}

float gurabo_bf16_to_f32(uint16_t bits)
{
    return f32_from_bits(gb_f32_bits_from_bf16(bits));
}
