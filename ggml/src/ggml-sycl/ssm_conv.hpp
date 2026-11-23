#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * add, ggml_tensor * silu_dst);

#define GGML_SYCL_CONV_WINDOW_MAX_SLOTS 8

// Match the recurrent window, rollback cache copies, convolution and SiLU.
struct ggml_sycl_conv_window_match {
    const ggml_tensor * concat   = nullptr;
    ggml_tensor *       conv     = nullptr;
    ggml_tensor *       add      = nullptr;
    ggml_tensor *       silu     = nullptr;
    int                 conv_idx = -1;
    int                 last_idx = -1;  // the silu node: the last node the kernel writes
    int                 n_slots  = 0;
    const ggml_tensor * cont[GGML_SYCL_CONV_WINDOW_MAX_SLOTS] = {};
    const ggml_tensor * cpy[GGML_SYCL_CONV_WINDOW_MAX_SLOTS]  = {};
};

// Purely structural, so graph_optimize and the compute loop see the same answer.
bool ggml_sycl_match_conv_window(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_conv_window_match & m);
void ggml_sycl_ssm_conv_window(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_window_match & m);

// out = res2 + scale*(res + conv), res2 optional; res, res2 and out are contiguous with the layout of dst
void ggml_sycl_ssm_conv_residual(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * res, float scale,
                                 const ggml_tensor * res2, ggml_tensor * out);

// One step of a short causal conv over a few tokens with its state, as one kernel (a work-item per channel and sequence):
//   cat  = CONCAT(past, x) along the window, written when cat is set
//   the state store: the last n_keep columns of cat (CPY) or K overlapping windows of them into slots (SET_ROWS)
//   out  = conv(cat, w), then scale*(res + out) + res2 when res is set
// past is [d_conv-1, C, S] with contiguous windows, x [n, C, S] with any strides, out / res / res2 [C, n, S] contiguous.
struct ggml_sycl_conv_step_args {
    const float * past;
    size_t        p_nb1, p_nb2;          // bytes
    const char *  x;
    size_t        x_nb0, x_nb1, x_nb2;   // bytes
    float *       cat;                   // [d_conv-1+n, C, S] contiguous, or null
    // store kind 1: CPY of cat[n + i0, c, s] (flattened i0 + n_keep*(c + C*s)) to a [ne0, ...] destination with byte strides
    //   st_nb0 / st_nb1 along its rows; kind 2: SET_ROWS of window j (cat[first + j + i0, c, s]) to row idx[j] of
    //   [n_keep, K, C, S] with byte strides st_nb0..st_nb3
    int             store;
    char *          st;
    int64_t         st_ne0;
    size_t          st_nb0, st_nb1, st_nb2, st_nb3;
    const int32_t * st_idx;
    int64_t         st_kw, st_first, n_keep;
    const float *   w;                   // [d_conv, C]
    float *         out;
    const float *   res;
    const float *   res2;
    float           scale;
    int64_t         C, n, S;
};

bool ggml_sycl_conv_step_supported(int64_t d_conv, int64_t n);
void ggml_sycl_conv_step(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_step_args & a);
