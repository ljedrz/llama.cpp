#pragma once

#include "common.hpp"

// prompts over a long F16 cache with the relative-position bias: flash attention on Xe2 XMX
bool ggml_sycl_fattn_xmx_supported(int device, const ggml_tensor * dst);
void ggml_sycl_fattn_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
