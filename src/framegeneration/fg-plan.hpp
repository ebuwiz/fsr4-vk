// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Frame Generation pass-plan reader (roadmap Phase 0 substrate -> Phase 1 runtime).
//
// The provider must never restate a number the pack already measures: the pack's OWN
// generated tables (tools/extract/frame-generation/plan.py -> pack/fg-plan.tsv,
// pack/pipelines.tsv, pack/fg-plan.json) are the single source for how many modules exist,
// which stage each belongs to, how each one is bound, and which intermediate render
// targets the plan needs. This reader turns those bytes into runtime records, and it
// reports a parse failure instead of guessing: an unreadable/unknown binding token marks
// the module `unresolved` and the substrate refuses that module rather than inventing a
// binding for it.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

namespace ffgplan {

// One de-duplicated (set,binding) slot of one module, in the exact grammar
// plan.py serializes and tools/framegeneration-exec.cpp parses
// ("set:binding:KIND" plus a trailing ":a" for a whole-array binding).
struct BindingGroup {
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bool arrayed = false;
    bool knownKind = false;  // an unknown/short token leaves the slot unresolved
};

// One compute module of the plan (474 of them on the pinned pack: 92 classic
// interpolation CS + 382 mlfi neural inference).
struct Module {
    std::string name;        // content addressed, "<container16>.spv"
    std::string stage;       // compute-classic | compute-neural
    std::string entry;       // mlfi_<tier>_passNNN for neural; CS for classic
    std::string transport;   // plain | collapsible
    std::uint32_t pass = 0;
    bool hasPass = false;
    std::uint32_t order = 0;
    std::uint32_t bindingCount = 0;
    std::array<std::uint32_t, 3> workgroup{};
    std::vector<BindingGroup> groups;
    bool resolved = false;
    std::uint32_t pushBytes = 0;  // from pipelines.tsv (every module declares 32)
};

// One intermediate render target the plan says the effect owns. The reference
// geometry (and byte count) come from the pack; followsUpscale is derived by
// comparing the reference extent against the plan's own reference_max_render /
// reference_max_upscale, so the runtime can resize the target to the host's
// real resolutions without hard-coding which roles are which.
struct IntermediateTarget {
    std::string role;
    std::uint32_t bytesPerTexel = 0;
    std::uint32_t refWidth = 0;
    std::uint32_t refHeight = 0;
    std::uint64_t refBytes = 0;
    bool followsUpscale = false;
};

// The plan's own summary facts (fg-plan.json), kept so the runtime can prove
// its parse agrees with the pack instead of trusting either one alone.
struct Facts {
    std::uint32_t computeModules = 0;
    std::uint32_t neural = 0;
    std::uint32_t classic = 0;
    std::uint32_t resolved = 0;
    std::uint32_t unresolved = 0;
    std::uint32_t unified = 0;
    std::uint64_t bindingGroups = 0;
    std::string orderSource;
    std::array<std::uint32_t, 2> referenceMaxRender{};
    std::array<std::uint32_t, 2> referenceMaxUpscale{};
    std::uint64_t referenceBytesTotal = 0;
    std::vector<IntermediateTarget> targets;
};

// A fully parsed plan. ok=false and reason set means the pack's own tables
// disagreed with each other or could not be parsed -- the provider then builds no
// substrate and keeps refusing, naming why.
struct Plan {
    std::vector<Module> modules;
    Facts facts;
    bool ok = false;
    std::string reason;

    std::vector<std::string> classicNames;  // stage==compute-classic, ordered
    std::vector<std::string> neuralNames;
};

// Parse the three tables. Each takes the bytes ffgassets::read already produced.
// Anything unparsable is reported in `reason` (never silently defaulted).
void parsePlanTsv(const std::vector<std::uint8_t>& text, Plan& out);
void parsePipelinesTsv(const std::vector<std::uint8_t>& text, Plan& out);
void parsePlanJson(const std::vector<std::uint8_t>& text, Plan& out);

// Cross-checks the module rows against the summary numbers (module counts, the
// distinct (set,binding) group total, resolved/unified tallies). Returns false with
// `reason` naming the first disagreement.
bool consistent(const Plan& plan, std::string& reason);

// Serialized binding census of one module in plan.py's grammar, so a
// provider-side record can be compared byte-for-byte against the pack's fg-plan.tsv
// (the same drift gate the executor uses, from the other side).
std::string groupsOf(const Module& module);

// The byte model of the plan's intermediates at host resolutions: returns the
// per-target bytes (host render/upscale extent x bytesPerTexel) plus the total,
// which is exactly what ffxQuery(GPU_MEMORY_USAGE) will report once allocated.
struct SizedTargets {
    struct Entry {
        std::string role;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint64_t bytes = 0;
    };
    std::vector<Entry> entries;
    std::uint64_t totalBytes = 0;
};

SizedTargets sizeTargets(const Facts& facts,
                         std::uint32_t renderWidth,
                         std::uint32_t renderHeight,
                         std::uint32_t upscaleWidth,
                         std::uint32_t upscaleHeight);

}  // namespace ffgplan

