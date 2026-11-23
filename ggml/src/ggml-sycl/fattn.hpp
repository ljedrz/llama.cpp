//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// Reports if flash attention runs this node with the oneMKL kernel, which waits on the host
bool ggml_sycl_flash_attn_ext_needs_sync(int device, const ggml_tensor * dst);

// Reports if flash attention runs this node with the oneDNN kernel
bool ggml_sycl_flash_attn_ext_uses_onednn(int device, const ggml_tensor * dst);

// Scratch that flash attention needs beyond the output tensor
struct ggml_sycl_fattn_extra {
    uintptr_t K_buffer_ptr     = 0;   // F16 copy of the K cache
    uintptr_t V_buffer_ptr     = 0;   // F16 copy of the V cache
    uintptr_t Q_buffer_ptr     = 0;   // dense F16 copy of Q, oneDNN only
    uintptr_t scale_buffer_ptr = 0;   // the softmax scale as an F16 scalar, oneDNN only
    uintptr_t out_buffer_ptr   = 0;   // F16 SDPA output before conversion to F32, oneDNN only
    uintptr_t end              = 0;   // one past the last reserved byte; sizes the allocation
};

// ggml_sycl_fattn_get_extra() is the single source of truth for the layout: it both sizes
// the reservation and hands out the pointers, so the two cannot disagree.
// Each field is the address of one reserved block, or 0 if that block was not reserved,
// in which case the caller allocates from the scratch pool instead.
ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(const ggml_tensor * dst);

size_t ggml_sycl_flash_attn_ext_get_alloc_size(const ggml_tensor * dst);

// the relative-position bias of a flash attention node (src[6..8]): bias[kv, q, h] = table[d, h, q] with
// d = min((uint32_t) (q_pos[q] - kv_pos[kv]), d_max), see ggml_flash_attn_ext_add_rel_bias
struct ggml_sycl_fattn_rel_bias {
    const char *    table  = nullptr;  // null: no bias
    const int32_t * kv_pos = nullptr;
    const int32_t * q_pos  = nullptr;
    int64_t         tb1 = 0, tb2 = 0;  // table strides in bytes (along the head and the query); distances are contiguous
    uint32_t        d_max = 0;         // n_dist - 1
};

// the bias table index of a query at position qp and a cell at position kp
static inline uint32_t ggml_sycl_rel_dist(int32_t qp, int32_t kp, uint32_t d_max) {
    const uint32_t d = (uint32_t) qp - (uint32_t) kp;
    return d < d_max ? d : d_max;
}

static inline ggml_sycl_fattn_rel_bias ggml_sycl_fattn_get_rel_bias(const ggml_tensor * dst) {
    ggml_sycl_fattn_rel_bias rel = {};
    if (const ggml_tensor * table = dst->src[6]) {
        rel.table  = (const char *) table->data;
        rel.kv_pos = (const int32_t *) dst->src[7]->data;
        rel.q_pos  = (const int32_t *) dst->src[8]->data;
        rel.tb1    = (int64_t) table->nb[1];
        rel.tb2    = (int64_t) table->nb[2];
        rel.d_max  = (uint32_t) (table->ne[0] - 1);
    }
    return rel;
}

void ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
