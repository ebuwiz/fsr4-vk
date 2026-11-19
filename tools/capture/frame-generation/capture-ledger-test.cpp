// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.
// GPU-free unit test for the capture ledger's bookkeeping (no Vulkan, no assets lib): it verifies pipeline
// registration, per-command-buffer last-bind, dispatch attribution, that an unbound / unregistered dispatch is
// IGNORED, that a repeat keeps the same pipeline, and the indirect (empty-dims) marker -> "-" in the emitted TSV.
#include "capture-ledger.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

int main() {
    using ffcap::Ledger;
    Ledger l;
    l.note_pipeline(1001, "a.spv");
    l.note_pipeline(1002, "b.spv");

    l.note_bind(7, 1001);
    l.note_dispatch(7, "8,8,1");
    l.note_bind(7, 1002);
    l.note_dispatch(7, "16,1,1");
    l.note_dispatch(7, "16,1,1");   // still bound to 1002 -> repeat recorded
    l.note_dispatch(9, "1,1,1");    // command buffer never bound -> ignored
    l.note_bind(8, 9999);
    l.note_dispatch(8, "4,4,4");    // pipeline never registered -> ignored
    l.note_bind(7, 1001);
    l.note_dispatch(7, "");         // indirect -> empty dims

    const std::vector<std::pair<std::string, std::string>> want = {
        {"a.spv", "8,8,1"}, {"b.spv", "16,1,1"}, {"b.spv", "16,1,1"}, {"a.spv", ""}};
    if (l.records() != want) {
        std::printf("FAIL records got=%zu want=%zu\n", l.records().size(), want.size());
        for (std::size_t i = 0; i < l.records().size(); ++i)
            std::printf("  [%zu] %s / %s\n", i, l.records()[i].first.c_str(), l.records()[i].second.c_str());
        return 1;
    }

    const std::string path = "/tmp/ffcap_ledger_test.tsv";
    l.write_tsv(path);
    std::ifstream in(path);
    std::vector<std::string> lines, wantLines = {
        "a.spv\t8,8,1", "b.spv\t16,1,1", "b.spv\t16,1,1", "a.spv\t-"};
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    if (lines != wantLines) {
        std::printf("FAIL tsv lines=%zu\n", lines.size());
        return 1;
    }
    std::printf("capture-ledger unit OK: %zu attributed records; unbound/unregistered ignored; indirect -> '-'\n",
                l.records().size());
    return 0;
}
