#include <gurabo/error.h>
#include <gurabo/device.h>
#include <gurabo/stream.h>
#include <gurabo/memory.h>
#include <gurabo/tensor.h>
#include <gurabo/ops.h>
#include <gurabo/gurabo.h>

int main()
{
    uint32_t count = 0;
    return gurabo_abi_version() == GURABO_ABI_VERSION &&
           gurabo_backend_count(&count) == GURABO_OK && count > 0 ? 0 : 1;
}
