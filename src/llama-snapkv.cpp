#include "llama-snapkv.h"

#include "ggml.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace {

struct snapkv_token_ref {
    uint32_t cell_idx;
    llama_pos pos;
    llama_kv_cell_ext ext;
};

bool env_flag_enabled(const char * value) {
    return value != nullptr && std::atoi(value) != 0;
}

std::vector<snapkv_token_ref> collect_tokens(const llama_kv_cells & cells, llama_seq_id seq_id) {
    std::vector<snapkv_token_ref> tokens;
    tokens.reserve(cells.get_used());

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i) || !cells.seq_has(i, seq_id)) {
            continue;
        }

        if (cells.seq_count(i) != 1) {
            return {};
        }

        tokens.push_back({
            /*.cell_idx =*/ i,
            /*.pos      =*/ cells.pos_get(i),
            /*.ext      =*/ cells.ext_get(i),
        });
    }

    std::sort(tokens.begin(), tokens.end(), [](const snapkv_token_ref & lhs, const snapkv_token_ref & rhs) {
        if (lhs.pos != rhs.pos) {
            return lhs.pos < rhs.pos;
        }

        return lhs.cell_idx < rhs.cell_idx;
    });

    return tokens;
}

} // namespace

llama_snapkv_params llama_snapkv_params_load_from_env() {
    llama_snapkv_params params;

    params.enabled = env_flag_enabled(std::getenv("LLAMA_SNAPKV"));

    if (const char * value = std::getenv("LLAMA_SNAPKV_WINDOW")) {
        params.window_size = std::max(1, std::atoi(value));
    }

    if (const char * value = std::getenv("LLAMA_SNAPKV_MAX_CAPACITY")) {
        params.max_capacity = std::max(1, std::atoi(value));
    }

    if (const char * value = std::getenv("LLAMA_SNAPKV_POOL_KERNEL")) {
        params.pooling_kernel = std::max(1, std::atoi(value));
    }

    if (params.max_capacity <= params.window_size) {
        params.enabled = false;
    }

    return params;
}

bool llama_snapkv_is_supported_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

bool llama_snapkv_should_schedule(
        const llama_snapkv_params & params,
        const llama_ubatch & ubatch,
        uint32_t n_stream,
        llama_swa_type swa_type) {
    if (!params.enabled) {
        return false;
    }

    if (n_stream != 1 || swa_type != LLAMA_SWA_TYPE_NONE) {
        return false;
    }

    if (ubatch.is_pos_2d()) {
        return false;
    }

    if (ubatch.n_seqs_unq != 1 || ubatch.n_tokens <= 1) {
        return false;
    }

    return true;
}

bool llama_snapkv_build_row_sets(
        const llama_snapkv_params & params,
        llama_seq_id seq_id,
        const llama_kv_cells & cells,
        std::vector<int32_t> & prefix_rows,
        std::vector<int32_t> & obs_rows) {
    prefix_rows.clear();
    obs_rows.clear();

    if (!params.enabled) {
        return false;
    }

    const auto tokens = collect_tokens(cells, seq_id);
    if (tokens.empty() || tokens.size() <= params.max_capacity || tokens.size() <= params.window_size) {
        return false;
    }

    const uint32_t n_tokens = tokens.size();
    const uint32_t obs_len = params.window_size;
    const uint32_t prefix_len = n_tokens - obs_len;
    const uint32_t prefix_keep = params.max_capacity - obs_len;

    if (prefix_len <= prefix_keep) {
        return false;
    }

    prefix_rows.reserve(prefix_len);
    obs_rows.reserve(obs_len);

    for (uint32_t idx = 0; idx < prefix_len; ++idx) {
        prefix_rows.push_back((int32_t) tokens[idx].cell_idx);
    }

    for (uint32_t idx = prefix_len; idx < n_tokens; ++idx) {
        obs_rows.push_back((int32_t) tokens[idx].cell_idx);
    }

    return true;
}

bool llama_snapkv_build_plan(
        const llama_snapkv_params & params,
        llama_seq_id seq_id,
        const llama_kv_cells & cells,
        const std::vector<int32_t> & selected_prefix_logical,
        llama_snapkv_plan & plan) {
    plan = {};

    if (!params.enabled) {
        return false;
    }

    const auto tokens = collect_tokens(cells, seq_id);
    if (tokens.empty() || tokens.size() <= params.max_capacity || tokens.size() <= params.window_size) {
        return false;
    }

    const uint32_t n_tokens = tokens.size();
    const uint32_t obs_len = params.window_size;
    const uint32_t prefix_len = n_tokens - obs_len;
    const uint32_t prefix_keep = params.max_capacity - obs_len;

    if (selected_prefix_logical.size() != prefix_keep) {
        return false;
    }

    plan.src_rows.reserve(params.max_capacity);
    plan.positions.reserve(params.max_capacity);
    plan.extents.reserve(params.max_capacity);

    for (int32_t idx_i32 : selected_prefix_logical) {
        if (idx_i32 < 0 || (uint32_t) idx_i32 >= prefix_len) {
            return false;
        }

        const uint32_t idx = (uint32_t) idx_i32;
        plan.src_rows.push_back(tokens[idx].cell_idx);
        plan.positions.push_back(tokens[idx].pos);
        plan.extents.push_back(tokens[idx].ext);
    }

    for (uint32_t idx = prefix_len; idx < n_tokens; ++idx) {
        plan.src_rows.push_back(tokens[idx].cell_idx);
        plan.positions.push_back(tokens[idx].pos);
        plan.extents.push_back(tokens[idx].ext);
    }

    return true;
}