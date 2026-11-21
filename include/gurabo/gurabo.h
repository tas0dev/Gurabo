#ifndef GURABO_H
#define GURABO_H
#include "device.h"
#include "error.h"
#include "memory.h"
#include "ops.h"
#include "stream.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GURABO_ABI_VERSION_MAJOR 1u
#define GURABO_ABI_VERSION_MINOR 5u
#define GURABO_ABI_VERSION ((GURABO_ABI_VERSION_MAJOR << 16) | GURABO_ABI_VERSION_MINOR)

uint32_t gurabo_abi_version(void);

#ifdef __cplusplus
}
#endif
#endif
