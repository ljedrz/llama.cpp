#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);
// out = res2 + scale*(res + conv), res2 optional; res, res2 and out are contiguous with the layout of dst
void ggml_sycl_ssm_conv_residual(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * res, float scale,
                                 const ggml_tensor * res2, ggml_tensor * out);
