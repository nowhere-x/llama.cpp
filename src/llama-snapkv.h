#pragma once

#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama.h"
#include "llama-kv-cells.h"

#include <vector>

struct llama_snapkv_params {
    bool enabled = false;
    uint32_t window_size = 32;
    uint32_t max_capacity = 512;
    uint32_t pooling_kernel = 5;
};

struct llama_snapkv_plan {
        std::vector<uint32_t> src_rows;
        std::vector<llama_pos> positions;
        std::vector<llama_kv_cell_ext> extents;
};

llama_snapkv_params llama_snapkv_params_load_from_env();

bool llama_snapkv_is_supported_type(ggml_type type);

bool llama_snapkv_should_schedule(
        const llama_snapkv_params & params,
        const llama_ubatch & ubatch,
        uint32_t n_stream,
        llama_swa_type swa_type);

bool llama_snapkv_build_row_sets(
        const llama_snapkv_params & params,
        llama_seq_id seq_id,
        const llama_kv_cells & cells,
        std::vector<int32_t> & prefix_rows,
        std::vector<int32_t> & obs_rows);

bool llama_snapkv_build_plan(
        const llama_snapkv_params & params,
        llama_seq_id seq_id,
        const llama_kv_cells & cells,
        const std::vector<int32_t> & selected_prefix_logical,
        llama_snapkv_plan & plan);