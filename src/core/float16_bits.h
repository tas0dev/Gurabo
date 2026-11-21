#ifndef GURABO_FLOAT16_BITS_H
#define GURABO_FLOAT16_BITS_H

#ifndef GB_KERNEL_INLINE
#define GB_KERNEL_INLINE static inline
#endif

GB_KERNEL_INLINE uint32_t gb_round_shift(uint32_t value, unsigned shift)
{
    uint32_t truncated = value >> shift;
    uint32_t midpoint = 1u << (shift - 1);
    uint32_t remainder = value & ((midpoint << 1) - 1);
    return truncated + (remainder > midpoint || (remainder == midpoint && (truncated & 1)));
}

GB_KERNEL_INLINE uint16_t gb_f16_from_f32_bits(uint32_t bits)
{
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
        return (uint16_t)(sign | gb_round_shift(fraction | 0x800000u, 126 - exponent));
    }
    /* Addition lets rounding carry into the exponent, including overflow. */
    return (uint16_t)(sign | (((exponent - 112) << 10) + gb_round_shift(fraction, 13)));
}

GB_KERNEL_INLINE uint32_t gb_f32_bits_from_f16(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    unsigned exponent = (bits >> 10) & 0x1Fu;
    uint32_t fraction = bits & 0x3FFu;
    if (exponent == 31) {
        return sign | 0x7F800000u | (fraction << 13);
    }
    if (exponent == 0) {
        if (!fraction) {
            return sign;
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
    return sign | ((uint32_t)exponent << 23) | (fraction << 13);
}

GB_KERNEL_INLINE uint16_t gb_bf16_from_f32_bits(uint32_t bits)
{
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return (uint16_t)((bits >> 16) | 0x0040u);
    }
    return (uint16_t)gb_round_shift(bits, 16);
}

GB_KERNEL_INLINE uint32_t gb_f32_bits_from_bf16(uint16_t bits)
{
    return (uint32_t)bits << 16;
}

#endif
