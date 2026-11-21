#include <stdio.h>
#include "core.h"

static _Thread_local gurabo_error_info thread_error;

uint32_t gurabo_abi_version(void)
{
    return GURABO_ABI_VERSION;
}

const char *gurabo_status_string(gurabo_status status)
{
    static const char *const messages[] = {
        "success", "invalid argument", "invalid handle", "out of range",
        "out of memory", "backend unavailable", "unsupported operation",
        "context mismatch", "resource busy", "not ready", "backend error",
        "device lost"
    };
    if (status < 0 || (size_t)status >= sizeof(messages) / sizeof(messages[0])) {
        return "unknown status";
    }
    return messages[status];
}

gurabo_error_info gb_make_error(gb_result result, gurabo_backend backend,
                                gurabo_error_phase phase, uint64_t submission)
{
    gurabo_error_info info = {0};
    info.struct_size = GURABO_ERROR_INFO_SIZE;
    info.status = result.status;
    info.phase = phase;
    info.backend = backend;
    info.native_domain = result.domain;
    info.native_code = result.code;
    info.submission_id = submission;
    snprintf(info.message, sizeof(info.message), "%s",
             result.message ? result.message : gurabo_status_string(result.status));
    return info;
}

gurabo_status gb_report(const gurabo_error_info *error)
{
    thread_error = *error;
    return error->status;
}

gurabo_status gb_call_result(gb_result result, gurabo_backend backend)
{
    if (result.status == GURABO_OK) {
        return GURABO_OK;
    }
    gurabo_error_info error = gb_make_error(result, backend, GURABO_ERROR_CALL, 0);
    return gb_report(&error);
}

gurabo_status gb_fail(gurabo_status status, gurabo_backend backend)
{
    return gb_call_result(gb_status(status), backend);
}

void gb_write_struct(void *out, const void *value, uint32_t requested, size_t available)
{
    /* Keep the caller's size so the same output object can be queried again. */
    size_t bytes = requested < available ? requested : available;
    memcpy(out, value, bytes);
    memcpy(out, &requested, sizeof(requested));
}

gurabo_status gb_get_error(const gurabo_error_info *error, gurabo_error_info *out)
{
    if (!out || out->struct_size < sizeof(uint32_t)) {
        return gb_fail(GURABO_INVALID_ARGUMENT, error->backend);
    }
    if (error->status == GURABO_OK) {
        return GURABO_NOT_READY;
    }
    gb_write_struct(out, error, out->struct_size, sizeof(*error));
    return GURABO_OK;
}

gurabo_status gurabo_get_thread_error(gurabo_error_info *out)
{
    return gb_get_error(&thread_error, out);
}
