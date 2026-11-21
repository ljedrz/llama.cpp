#ifndef GGML_SYCL_MMVF_HPP
#define GGML_SYCL_MMVF_HPP

#include "common.hpp"

// MUL_MAT of F32 or BF16 weights with up to 8 src1 columns, and of F16 KV cache views for one token of
// grouped-query attention, one sub-group per dst row.
// src0 must not be in a split buffer. Returns false when the case is not handled (types, layout or width).
bool ggml_sycl_mul_mat_vec_f(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst);

#endif // GGML_SYCL_MMVF_HPP
