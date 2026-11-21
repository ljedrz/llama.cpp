#ifndef GGML_SYCL_MMVF_HPP
#define GGML_SYCL_MMVF_HPP

#include "common.hpp"

// MUL_MAT of F32 or BF16 weights with up to 8 src1 columns, and of F16 KV cache views for one token of
// grouped-query attention, one sub-group per dst row.
// src0 must not be in a split buffer. Returns false when the case is not handled (types, layout or width).
bool ggml_sycl_mul_mat_vec_f(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst);

// dst = (x - tanh(wf.g)*x) + tanh(wu.g)*m for F32 weights wf and wu of nrows rows of k values, over ncols columns.
// Rows are 16-byte aligned, the column strides are in bytes.
struct ggml_sycl_gate_mix_args {
    const char * wf;
    const char * wu;
    const char * g;
    const char * x;
    const char * m;
    char *       dst;
    int64_t      k, nrows, ncols;
    size_t       nb_w, nb_g, nb_x, nb_m, nb_d;
};

bool ggml_sycl_gate_mix_f32_supported(int64_t k, int64_t ncols);
void ggml_sycl_gate_mix_f32(const ggml_sycl_gate_mix_args & a, dpct::queue_ptr stream);

#endif // GGML_SYCL_MMVF_HPP
