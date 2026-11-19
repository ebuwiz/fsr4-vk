// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.
//
// capture-ledger.hpp - the Vulkan-free bookkeeping core of the capture shim. Kept header-only and free of
// every Vulkan type (handles are opaque std::uintptr_t), so it links + runs GPU-free (see capture-ledger-test.cpp)
// and does NOT depend on the runtime assets lib. The shim (capture-shim.cpp) casts real VkPipeline / VkCommand
// Buffer handles to these integers and feeds them here. The ledger records, per vkCmdDispatch* call, the SPIR-V
// file the bound pipeline was CREATED with (its identity) and the threadgroup count (or an "indirect" marker,
// whose args live on the GPU), in call ORDER. It writes a TSV / JSON exactly as capture-diff.py consumes.
//
// Identity policy: the ledger NEVER computes a hash and NEVER invents an order. It only says "this dispatch
// used THIS pipeline, whose SPIR-V I dumped at THIS path, for THIS threadgrid". capture-diff.py then hashes that
// file and matches it to the pack plan. The ORDER is therefore AMD's, observed at runtime.
#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ffcap {

class Ledger {
public:
    // Register a pipeline's dumped SPIR-V file at pipeline-create time.
    void note_pipeline(std::uintptr_t pipeline, const std::string& spv_path) {
        pipeline_spv_[pipeline] = spv_path;
    }
    // Remember the last compute pipeline bound per command buffer (VkCmdBindPipeline).
    void note_bind(std::uintptr_t command_buffer, std::uintptr_t pipeline) {
        bound_[command_buffer] = pipeline;
    }
    // Record one dispatch (dims = "x,y,z", or the empty string -> vkCmdDispatchIndirect with GPU-side args).
    void note_dispatch(std::uintptr_t command_buffer, const std::string& dims) {
        auto b = bound_.find(command_buffer);
        if (b == bound_.end()) {
            return;  // dispatch before any bind: nothing to attribute, keep the trace clean
        }
        auto spv = pipeline_spv_.find(b->second);
        if (spv == pipeline_spv_.end()) {
            return;  // bound pipeline was never registered (not a compute pipeline we captured)
        }
        records_.emplace_back(spv->second, dims);
    }

    const std::vector<std::pair<std::string, std::string>>& records() const { return records_; }

    void write_tsv(const std::string& path) const {
        std::ofstream out(path);
        for (const auto& r : records_) {
            out << r.first << '\t' << (r.second.empty() ? "-" : r.second) << '\n';
        }
    }
    void write_json(const std::string& path) const {
        std::ofstream out(path);
        out << "{\n  \"dispatches\": [\n";
        for (std::size_t i = 0; i < records_.size(); ++i) {
            out << "    {\"spirv_ref\": \"" << records_[i].first << "\", \"dims\": \""
                << (records_[i].second.empty() ? "-" : records_[i].second) << "\"}";
            out << (i + 1 < records_.size() ? ",\n" : "\n");
        }
        out << "  ]\n}\n";
    }

private:
    std::unordered_map<std::uintptr_t, std::string> pipeline_spv_;
    std::unordered_map<std::uintptr_t, std::uintptr_t> bound_;
    std::vector<std::pair<std::string, std::string>> records_;
};

} // namespace ffcap
