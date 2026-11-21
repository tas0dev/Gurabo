#include <float.h>
#include <stdint.h>
#include <string.h>
#include <gurabo/tensor.h>

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

static uint32_t round_shift(uint32_t value, unsigned shift)
{
    uint32_t truncated = value >> shift;
    uint32_t midpoint = UINT32_C(1) << (shift - 1);
    uint32_t remainder = value & ((midpoint << 1) - 1);
    return truncated + (remainder > midpoint || (remainder == midpoint && (truncated & 1)));
}

uint16_t gurabo_f16_from_f32(float value)
{
    uint32_t bits = f32_bits(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    unsigned exponent = (bits >> 23) & 0xFFu;
    uint32_t fraction = bits & 0x7FFFFFu;
    if (exponent == 255 && fraction) {
        return (uint16_t)(sign | 0x7E00u | (fraction >> 13));
    }
    if (exponent >= 143) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exponent < 102) {
        return (uint16_t)sign;
    }
    if (exponent < 113) {
        return (uint16_t)(sign | round_shift(fraction | 0x800000u, 126 - exponent));
    }
    /* Addition lets rounding carry into the exponent, including overflow. */
    return (uint16_t)(sign | (((exponent - 112) << 10) + round_shift(fraction, 13)));
}

float gurabo_f16_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    unsigned exponent = (bits >> 10) & 0x1Fu;
    uint32_t fraction = bits & 0x3FFu;
    if (exponent == 31) {
        return f32_from_bits(sign | 0x7F800000u | (fraction << 13));
    }
    if (exponent == 0) {
        if (!fraction) {
            return f32_from_bits(sign);
        }
        exponent = 113;
        while (!(fraction & 0x400u)) {
            fraction <<= 1;
            --exponent;
        }
        fraction &= 0x3FFu;
    } else {
        exponent += 112;
    }
    return f32_from_bits(sign | ((uint32_t)exponent << 23) | (fraction << 13));
}

uint16_t gurabo_bf16_from_f32(float value)
{
    uint32_t bits = f32_bits(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (uint16_t)((bits >> 16) | 0x0040u);
    }
    return (uint16_t)round_shift(bits, 16);
}

float gurabo_bf16_to_f32(uint16_t bits)
{
    return f32_from_bits((uint32_t)bits << 16);
}
