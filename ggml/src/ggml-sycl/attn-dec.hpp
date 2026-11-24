#ifndef GGML_SYCL_ATTN_DEC_HPP
#define GGML_SYCL_ATTN_DEC_HPP

#include "common.hpp"

// attention of a few tokens over an F16 KV cache without flash attention: MUL_MAT(K, q) -> SOFT_MAX(mask, sinks) ->
// MUL_MAT(V^T, p) as one pass over K and V per KV head, for the n_tok*gqa query vectors that share a KV head
struct ggml_sycl_attn_dec_args {
    const char * q;      // F32 [D, n_tok, n_head]
    const char * k;      // F16 [D, n_kv, n_head_kv]
    const char * v;      // F16 V^T [n_kv, D, n_head_kv], or with v_rows V [D, n_kv, n_head_kv] (flash attention layout)
    const char * mask;   // F32 or F16 [n_kv, >= n_tok, 1 or n_head], or null
    const float * sinks; // [n_head] or null
    float *      dst;    // F32, element (d, token, head) at d + token*o_t + head*o_h
    int64_t      D, n_kv, n_tok, n_head, n_head_kv, mask_heads;
    size_t       q_nb1, q_nb2;
    size_t       k_nb1, k_nb2;
    size_t       v_nb1, v_nb2;
    size_t       m_nb0, m_nb1, m_nb2;
    float        scale;
    bool         mask_f16;
    bool         v_rows;
    int64_t      o_t, o_h;
    // indexed bias (see ggml_flash_attn_ext_add_indexed_bias): table[idx[kv, token], head, token], or null
    const char *    rel_table;
    const int32_t * rel_idx;
    size_t          r_tb1, r_tb2;   // table strides in bytes along the head and the token
    size_t          r_ib1;          // idx row stride in elements
};

bool ggml_sycl_attn_dec_supported(int64_t D, int64_t nq);

void ggml_sycl_attn_dec(ggml_backend_sycl_context & ctx, const ggml_sycl_attn_dec_args & a);

#endif // GGML_SYCL_ATTN_DEC_HPP
