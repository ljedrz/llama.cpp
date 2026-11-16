// Fusions for the elementwise tails of the Mach-1 payload graph (qwen4exp): the routed-expert weighted sum, the
// paired gate + up + SWIGLU products at decode, and the hc activation tails. Each fused kernel computes the same fp32
// operations in the same order as the nodes it replaces (the mach1 sources build with -fp-model=precise).

#include "mach1.hpp"

#include <cstring>
#include <initializer_list>

#define MACH1_FUSE_WG 256

static bool mach1_f32c(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static bool mach1_unary_is(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

// a SCALE with zero bias, returning its factor
static bool mach1_scale_of(const ggml_tensor * t, float * s) {
    if (t->op != GGML_OP_SCALE) {
        return false;
    }
    float b;
    memcpy(s, (const float *) t->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) t->op_params + 1, sizeof(float));
    return b == 0.0f;
}

// the fused kernel reads `src` at other indices than it writes `dst`: they must not overlap
static bool mach1_disjoint(const ggml_tensor * dst, const ggml_tensor * src) {
    const char * a0 = (const char *) dst->data;
    const char * a1 = a0 + ggml_nbytes(dst);
    const char * b0 = (const char *) src->data;
    const char * b1 = b0 + ggml_nbytes(src);
    return a1 <= b0 || b1 <= a0;
}

// elementwise reads of `src` at the index being written are safe when it is exactly `dst`
static bool mach1_same_or_disjoint(const ggml_tensor * dst, const ggml_tensor * src) {
    return dst->data == src->data || mach1_disjoint(dst, src);
}

// Where a fused kernel writes. The graph allocator may place the output over an input that dies
// inside the fused subgraph (the nodes it replaces would have run in order); the fused kernel reads
// its inputs at other indices than it writes, so then it writes to a temporary that is copied into
// place afterwards (on the same in-order queue) instead of giving up the fusion. `strict` inputs
// must be disjoint from the output, `elementwise` ones may also be exactly the output.
struct mach1_fuse_out {
    ggml_sycl_pool_alloc<float> tmp;
    float * dst;
    float * ptr;
    size_t  n;

    mach1_fuse_out(ggml_backend_sycl_context & ctx, ggml_tensor * out, std::initializer_list<const ggml_tensor *> strict,
                   std::initializer_list<const ggml_tensor *> elementwise = {}) :
        tmp(ctx.pool()), dst((float *) out->data), ptr((float *) out->data), n((size_t) ggml_nelements(out)) {
        bool ok = true;
        for (const ggml_tensor * t : strict) {
            ok = ok && mach1_disjoint(out, t);
        }
        for (const ggml_tensor * t : elementwise) {
            ok = ok && mach1_same_or_disjoint(out, t);
        }
        if (!ok) {
            ptr = tmp.alloc(n);
        }
    }

    void finish(dpct::queue_ptr stream) {
        if (ptr != dst) {
            stream->memcpy(dst, ptr, n*sizeof(float));
        }
    }
};

static float mach1_sigmoid(float x) {
    return 1.0f/(1.0f + sycl::exp(-x));
}


//
// activation tails: SCALE -> SIGMOID -> SCALE (the hc inject) and SCALE -> SILU (the hc down)
//
static int mach1_fuse_scale_act(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 2 > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * s1 = cgraph->nodes[i];
    ggml_tensor * a  = cgraph->nodes[i + 1];
    float f1, f2 = 1.0f;
    if (!mach1_scale_of(s1, &f1) || a->src[0] != s1 || !mach1_f32c(s1) || !mach1_f32c(a) || !mach1_f32c(s1->src[0])) {
        return 0;
    }
    const bool silu = mach1_unary_is(a, GGML_UNARY_OP_SILU);
    const bool sig  = mach1_unary_is(a, GGML_UNARY_OP_SIGMOID);
    if (!silu && !sig) {
        return 0;
    }
    ggml_tensor * out = a;
    int count = 2;
    if (sig && i + 3 <= cgraph->n_nodes) {
        ggml_tensor * s2 = cgraph->nodes[i + 2];
        if (mach1_scale_of(s2, &f2) && s2->src[0] == a && mach1_f32c(s2)) {
            const ggml_op ops3[3] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE };
            const int     outs3[1] = { i + 2 };
            if (ggml_can_fuse_subgraph(cgraph, i, 3, ops3, outs3, 1)) {
                out   = s2;
                count = 3;
            } else {
                f2 = 1.0f;
            }
        } else {
            f2 = 1.0f;
        }
    }
    if (count == 2) {
        const ggml_op ops2[2] = { GGML_OP_SCALE, GGML_OP_UNARY };
        const int     outs2[1] = { i + 1 };
        if (!ggml_can_fuse_subgraph(cgraph, i, 2, ops2, outs2, 1)) {
            return 0;
        }
    }
    const ggml_tensor * src = s1->src[0];
    if (ggml_nelements(out) != ggml_nelements(src)) {
        return 0;
    }
    mach1_fuse_out fo(ctx, out, {}, { src });
    const float * xd = (const float *) src->data;
    float       * yd = fo.ptr;
    const int64_t total = ggml_nelements(src);
    const bool    tail  = count == 3;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t)((total + MACH1_FUSE_WG - 1)/MACH1_FUSE_WG)*MACH1_FUSE_WG),
                          sycl::range<1>(MACH1_FUSE_WG)),
        [=](sycl::nd_item<1> it) {
            const int64_t k = (int64_t) it.get_global_id(0);
            if (k >= total) {
                return;
            }
            const float x = xd[k]*f1;
            float y;
            if (silu) {
                y = x/(1.0f + sycl::exp(-x));
            } else {
                y = mach1_sigmoid(x);
            }
            yd[k] = tail ? y*f2 : y;
        });
    fo.finish(ctx.stream());
    return count - 1;
}


//
// routed-expert weighted sum: out = sum_s y_s * w_s   (MUL, VIEW x k, ADD x (k - 1))
//
static int mach1_fuse_moe_wsum(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * mul = cgraph->nodes[i];
    if (mul->op != GGML_OP_MUL || mul->src[0]->op != GGML_OP_MACH1_D4_MM) {
        return 0;
    }
    const ggml_tensor * y = mul->src[0];
    const ggml_tensor * w = mul->src[1];
    const int64_t n  = y->ne[0];
    const int64_t k  = y->ne[1];
    const int64_t nt = y->ne[2];
    const int count = 1 + (int) k + (int) (k - 1);
    if (k < 2 || count >= 32 || i + count > cgraph->n_nodes || !mach1_f32c(y) || !mach1_f32c(mul) ||
        !ggml_are_same_shape(y, mul) || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
        w->ne[0] != 1 || w->ne[1] != k || w->ne[2] != nt) {
        return 0;
    }
    ggml_op ops[32];
    ops[0] = GGML_OP_MUL;
    for (int s = 0; s < k; ++s) {
        const ggml_tensor * v = cgraph->nodes[i + 1 + s];
        if (v->op != GGML_OP_VIEW || v->src[0] != mul || v->ne[0] != n || v->ne[1] != nt ||
            v->nb[1] != mul->nb[2] || (const char *) v->data != (const char *) mul->data + s*mul->nb[1]) {
            return 0;
        }
        ops[1 + s] = GGML_OP_VIEW;
    }
    const ggml_tensor * prev = cgraph->nodes[i + 1];
    for (int s = 1; s < k; ++s) {
        const ggml_tensor * a = cgraph->nodes[i + (int) k + s];
        if (a->op != GGML_OP_ADD || a->src[0] != prev || a->src[1] != cgraph->nodes[i + 1 + s] || a->type != GGML_TYPE_F32) {
            return 0;
        }
        ops[k + s] = GGML_OP_ADD;
        prev = a;
    }
    ggml_tensor * out = cgraph->nodes[i + count - 1];
    const int outs[1] = { i + count - 1 };
    if (!mach1_f32c(out) || ggml_nelements(out) != n*nt || !ggml_can_fuse_subgraph(cgraph, i, count, ops, outs, 1)) {
        return 0;
    }
    mach1_fuse_out fo(ctx, out, { y, w });
    const float * yd = (const float *) y->data;
    const float * wd = (const float *) w->data;
    float       * od = fo.ptr;
    const int64_t total = n*nt;
    const int     K     = (int) k;
    ctx.stream()->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t)((total + MACH1_FUSE_WG - 1)/MACH1_FUSE_WG)*MACH1_FUSE_WG),
                          sycl::range<1>(MACH1_FUSE_WG)),
        [=](sycl::nd_item<1> it) {
            const int64_t q = (int64_t) it.get_global_id(0);
            if (q >= total) {
                return;
            }
            const int64_t j = q % n;
            const int64_t t = q / n;
            const float * yt = yd + t*n*K + j;
            const float * wt = wd + t*K;
            float acc = yt[0]*wt[0];
            for (int s = 1; s < K; ++s) {
                const float p = yt[(int64_t) s*n]*wt[s];
                acc = acc + p;
            }
            od[q] = acc;
        });
    fo.finish(ctx.stream());
    return count - 1;
}

//
// gate, up, SWIGLU(gate, up) as one paired pass at decode: the routed experts (D4) and the
// shared expert (RT)
//
template <ggml_op OP>
static int mach1_fuse_pair(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 3 > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * g   = cgraph->nodes[i];
    ggml_tensor * u   = cgraph->nodes[i + 1];
    ggml_tensor * glu = cgraph->nodes[i + 2];
    if (g->op != OP || u->op != OP || glu->op != GGML_OP_GLU ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
        glu->src[0] != g || glu->src[1] != u || !ggml_are_same_shape(g, u) || !ggml_are_same_shape(g, glu)) {
        return 0;
    }
    const ggml_op ops[3]  = { OP, OP, GGML_OP_GLU };
    const int     outs[1] = { i + 2 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 3, ops, outs, 1)) {
        return 0;
    }
    // the last pass reads only the pass's own scratch, so out may overlap anything
    const bool done = OP == GGML_OP_MACH1_D4_MM ? ggml_sycl_mach1_d4_pair_swiglu(ctx, g, u, glu)
                                                : ggml_sycl_mach1_rt_pair_swiglu(ctx, g, u, glu);
    return done ? 2 : 0;
}

int ggml_sycl_mach1_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = getenv("GGML_SYCL_MACH1_FUSE") == nullptr || atoi(getenv("GGML_SYCL_MACH1_FUSE")) != 0;
    if (!on) {
        return 0;
    }
    switch (cgraph->nodes[i]->op) {
        case GGML_OP_SCALE:       return mach1_fuse_scale_act(ctx, cgraph, i);
        case GGML_OP_MUL:         return mach1_fuse_moe_wsum(ctx, cgraph, i);
        case GGML_OP_MACH1_D4_MM: return mach1_fuse_pair<GGML_OP_MACH1_D4_MM>(ctx, cgraph, i);
        case GGML_OP_MACH1_RT_MM: return mach1_fuse_pair<GGML_OP_MACH1_RT_MM>(ctx, cgraph, i);
        default:                  return 0;
    }
}
