// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "fg-plan.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ffgplan {

namespace {

std::vector<std::string> splitLines(const std::vector<std::uint8_t>& bytes) {
    std::string text(bytes.begin(), bytes.end());
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(text.substr(start));
            break;
        }
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return lines;
}

std::vector<std::string> splitTabs(const std::string& line) {
    std::vector<std::string> cols;
    std::size_t start = 0;
    while (true) {
        const auto end = line.find('\t', start);
        if (end == std::string::npos) {
            cols.push_back(line.substr(start));
            break;
        }
        cols.push_back(line.substr(start, end - start));
        start = end + 1;
    }
    return cols;
}

std::vector<std::string> splitOn(const std::string& text, char sep) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (true) {
        const auto end = text.find(sep, start);
        if (end == std::string::npos) {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return parts;
}

VkDescriptorType kindOf(const std::string& token, bool& known) {
    known = true;
    if (token == "SB") return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if (token == "UB") return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    if (token == "CIS") return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    if (token == "SI") return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (token == "SMP") return VK_DESCRIPTOR_TYPE_SAMPLER;
    known = false;
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

const char* kindName(VkDescriptorType type) {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "SB";
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UB";
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "CIS";
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "SI";
        case VK_DESCRIPTOR_TYPE_SAMPLER: return "SMP";
        default: return "U";
    }
}

std::uint32_t u32Of(const std::string& text) {
    return text.empty() ? 0u : static_cast<std::uint32_t>(std::stoul(text));
}

std::uint64_t u64Of(const std::string& text) {
    return text.empty() ? 0ull : static_cast<std::uint64_t>(std::stoull(text));
}

}  // namespace

// Parses plan.py's binding census ("0:0:CIS:a;5:0:UB"). A token without a known
// kind leaves the module unresolved: the substrate then refuses that module instead
// of inventing a binding the pack never declared.
static std::vector<BindingGroup> parseGroups(const std::string& text, bool& allKnown) {
    std::vector<BindingGroup> groups;
    allKnown = true;
    for (const auto& token : splitOn(text, ';')) {
        if (token.empty()) {
            continue;
        }
        const auto parts = splitOn(token, ':');
        if (parts.size() < 3) {
            allKnown = false;
            continue;
        }
        BindingGroup g;
        g.set = u32Of(parts[0]);
        g.binding = u32Of(parts[1]);
        bool known = false;
        g.type = kindOf(parts[2], known);
        g.knownKind = known;
        if (!known) {
            allKnown = false;
        }
        g.arrayed = parts.size() > 3 && parts[3] == "a";
        groups.push_back(g);
    }
    return groups;
}

static std::string joinGroups(const std::vector<BindingGroup>& groups) {
    std::string out;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const auto& g = groups[i];
        if (i != 0) {
            out += ';';
        }
        out += std::to_string(g.set) + ':' + std::to_string(g.binding) + ':' +
               (g.knownKind ? kindName(g.type) : std::string("U"));
        if (g.arrayed) {
            out += ":a";
        }
    }
    return out;
}

std::string groupsOf(const Module& module) {
    return joinGroups(module.groups);
}

void parsePlanTsv(const std::vector<std::uint8_t>& text, Plan& out) {
    const auto lines = splitLines(text);
    if (lines.size() < 2) {
        out.reason = "fg-plan.tsv has no rows";
        return;
    }
    const auto header = splitTabs(lines[0]);
    auto column = [&](const std::string& wanted) -> std::size_t {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == wanted) {
                return i;
            }
        }
        return std::string::npos;
    };
    const std::size_t nameCol = column("name");
    const std::size_t stageCol = column("stage");
    const std::size_t groupsCol = column("groups");
    const std::size_t workCol = column("w0,w1,w2");
    const std::size_t orderCol = column("order");
    const std::size_t entryCol = column("entry");
    const std::size_t passCol = column("pass");
    const std::size_t countCol = column("binding_count");
    const std::size_t resolvedCol = column("resolved");
    const std::size_t transportCol = column("transport");
    if (nameCol == std::string::npos || stageCol == std::string::npos ||
        groupsCol == std::string::npos || workCol == std::string::npos) {
        out.reason = "fg-plan.tsv header lacks name/stage/groups/w0,w1,w2";
        return;
    }
    const std::size_t widest = std::max({nameCol, stageCol, groupsCol, workCol});
    for (std::size_t row = 1; row < lines.size(); ++row) {
        const auto cols = splitTabs(lines[row]);
        if (cols.size() <= widest) {
            out.reason = "fg-plan.tsv row " + std::to_string(row) + " is short";
            return;
        }
        Module m;
        m.name = cols[nameCol];
        m.stage = cols[stageCol];
        if (entryCol != std::string::npos) m.entry = cols[entryCol];
        if (transportCol != std::string::npos) m.transport = cols[transportCol];
        if (orderCol != std::string::npos) m.order = u32Of(cols[orderCol]);
        if (passCol != std::string::npos && !cols[passCol].empty()) {
            m.pass = u32Of(cols[passCol]);
            m.hasPass = true;
        }
        if (countCol != std::string::npos) m.bindingCount = u32Of(cols[countCol]);
        m.resolved = resolvedCol == std::string::npos ? true
                                                      : cols[resolvedCol] == "1";
        bool allKnown = true;
        m.groups = parseGroups(cols[groupsCol], allKnown);
        if (countCol != std::string::npos && m.groups.size() != m.bindingCount) {
            m.resolved = false;
        }
        if (!allKnown) {
            m.resolved = false;
        }
        const auto parts = splitOn(cols[workCol], ',');
        for (std::size_t i = 0; i < 3 && i < parts.size(); ++i) {
            m.workgroup[i] = u32Of(parts[i]);
        }
        if (m.workgroup[0] == 0u || m.workgroup[1] == 0u || m.workgroup[2] == 0u) {
            m.resolved = false;
        }
        out.modules.push_back(std::move(m));
    }
}

void parsePipelinesTsv(const std::vector<std::uint8_t>& text, Plan& out) {
    // pipelines.tsv: name stage entry push_bytes collisions standardizable layout transport
    // Attaches the push-constant byte count (and the layout column as a cross-check)
    // to the module records already parsed from fg-plan.tsv.
    const auto lines = splitLines(text);
    if (lines.size() < 2) {
        out.reason = "pipelines.tsv has no rows";
        return;
    }
    const auto header = splitTabs(lines[0]);
    auto column = [&](const std::string& wanted) -> std::size_t {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == wanted) return i;
        }
        return std::string::npos;
    };
    const std::size_t nameCol = column("name");
    const std::size_t pushCol = column("push_bytes");
    const std::size_t layoutCol = column("layout");
    if (nameCol == std::string::npos || pushCol == std::string::npos) {
        out.reason = "pipelines.tsv header lacks name/push_bytes";
        return;
    }
    std::unordered_map<std::string, Module*> index;
    for (auto& m : out.modules) {
        index[m.name] = &m;
    }
    for (std::size_t row = 1; row < lines.size(); ++row) {
        const auto cols = splitTabs(lines[row]);
        if (cols.size() <= std::max(nameCol, pushCol)) {
            out.reason = "pipelines.tsv row " + std::to_string(row) + " is short";
            return;
        }
        const auto found = index.find(cols[nameCol]);
        if (found == index.end()) {
            continue;  // a non-compute container (VS/PS/PSV) has no plan row by design
        }
        found->second->pushBytes = u32Of(cols[pushCol]);
        if (layoutCol != std::string::npos) {
            // The two columns are NOT the same view and must not be compared as text:
            // pipelines.tsv's layout lists EVERY descriptor variable (AMD's HLSL re-binds
            // one heap slot as several same-kind views), while fg-plan.tsv's groups column
            // is the DE-DUPLICATED census the layouts are built from. What can be checked
            // across both is the KIND of each (set,binding) slot: a slot the two tables
            // disagree about, or one claimed by two DIFFERENT kinds, is drift -> the
            // module becomes unusable rather than being bound with a guess.
            std::unordered_map<std::uint64_t, VkDescriptorType> slotKinds;
            bool allKnown = true;
            bool crossKind = false;
            for (const auto& group : parseGroups(cols[layoutCol], allKnown)) {
                const auto key =
                    (static_cast<std::uint64_t>(group.set) << 32u) |
                    static_cast<std::uint64_t>(group.binding);
                const auto foundSlot = slotKinds.find(key);
                if (foundSlot == slotKinds.end()) {
                    slotKinds[key] = group.type;
                } else if (foundSlot->second != group.type) {
                    crossKind = true;
                }
            }
            bool drift = !allKnown || crossKind;
            for (const auto& group : found->second->groups) {
                const auto key =
                    (static_cast<std::uint64_t>(group.set) << 32u) |
                    static_cast<std::uint64_t>(group.binding);
                const auto slot = slotKinds.find(key);
                if (slot == slotKinds.end() || slot->second != group.type) {
                    drift = true;
                    break;
                }
            }
            if (drift) {
                found->second->resolved = false;
            }
        }
    }
}

namespace {

std::uint64_t jsonNumber(const std::string& t, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    const auto pos = t.find(pat);
    if (pos == std::string::npos) {
        return 0;
    }
    auto colon = t.find(':', pos + pat.size());
    if (colon == std::string::npos) {
        return 0;
    }
    ++colon;
    while (colon < t.size() && (t[colon] == ' ' || t[colon] == '\n' || t[colon] == '\t')) {
        ++colon;
    }
    std::size_t end = colon;
    while (end < t.size() && t[end] >= '0' && t[end] <= '9') {
        ++end;
    }
    if (end == colon) {
        return 0;
    }
    return u64Of(t.substr(colon, end - colon));
}

std::string jsonStringValue(const std::string& t, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    const auto pos = t.find(pat);
    if (pos == std::string::npos) {
        return {};
    }
    const auto colon = t.find(':', pos + pat.size());
    if (colon == std::string::npos) {
        return {};
    }
    const auto open = t.find('"', colon + 1);
    if (open == std::string::npos) {
        return {};
    }
    const auto close = t.find('"', open + 1);
    if (close == std::string::npos) {
        return {};
    }
    return t.substr(open + 1, close - open - 1);
}

std::array<std::uint32_t, 2> jsonPair(const std::string& t, const std::string& key) {
    std::array<std::uint32_t, 2> out{};
    const std::string pat = "\"" + key + "\"";
    const auto pos = t.find(pat);
    if (pos == std::string::npos) {
        return out;
    }
    const auto open = t.find('[', pos + pat.size());
    if (open == std::string::npos) {
        return out;
    }
    const auto close = t.find(']', open + 1);
    if (close == std::string::npos) {
        return out;
    }
    const auto parts = splitOn(t.substr(open + 1, close - open - 1), ',');
    for (std::size_t i = 0; i < 2 && i < parts.size(); ++i) {
        out[i] = u32Of(parts[i]);
    }
    return out;
}

}  // namespace

void parsePlanJson(const std::vector<std::uint8_t>& bytes, Plan& out) {
    const std::string t(bytes.begin(), bytes.end());
    Facts& f = out.facts;
    f.computeModules = static_cast<std::uint32_t>(jsonNumber(t, "compute_modules"));
    f.neural = static_cast<std::uint32_t>(jsonNumber(t, "neural"));
    f.classic = static_cast<std::uint32_t>(jsonNumber(t, "classic"));
    f.bindingGroups = jsonNumber(t, "distinct_binding_groups_total");
    f.resolved = static_cast<std::uint32_t>(jsonNumber(t, "resolved"));
    f.unresolved = static_cast<std::uint32_t>(jsonNumber(t, "unresolved"));
    f.unified = static_cast<std::uint32_t>(jsonNumber(t, "unified"));
    f.referenceBytesTotal = jsonNumber(t, "intermediate_bytes_total");
    f.orderSource = jsonStringValue(t, "order_source");
    f.referenceMaxRender = jsonPair(t, "reference_max_render");
    f.referenceMaxUpscale = jsonPair(t, "reference_max_upscale");
    const std::string section = "\"intermediate_targets\"";
    const auto pos = t.find(section);
    if (pos == std::string::npos) {
        out.reason = "fg-plan.json has no intermediate_targets";
        return;
    }
    const auto arrayOpen = t.find('[', pos + section.size());
    const auto arrayClose = arrayOpen == std::string::npos
                                 ? std::string::npos
                                 : t.find(']', arrayOpen + 1);
    if (arrayOpen == std::string::npos || arrayClose == std::string::npos) {
        out.reason = "fg-plan.json intermediate_targets is not an array";
        return;
    }
    std::size_t cursor = arrayOpen + 1;
    while (true) {
        const auto open = t.find('{', cursor);
        if (open == std::string::npos || open > arrayClose) {
            break;
        }
        const auto close = t.find('}', open + 1);
        if (close == std::string::npos || close > arrayClose) {
            break;
        }
        const auto block = t.substr(open, close - open + 1);
        IntermediateTarget target;
        target.role = jsonStringValue(block, "role");
        target.refBytes = jsonNumber(block, "bytes");
        target.bytesPerTexel = static_cast<std::uint32_t>(
            jsonNumber(block, "bytes_per_texel"));
        target.refWidth = static_cast<std::uint32_t>(jsonNumber(block, "width"));
        target.refHeight = static_cast<std::uint32_t>(jsonNumber(block, "height"));
        // Which geometry a target follows is derived from the plan's own reference
        // extents, never from the role name: an extent equal to the plan's max
        // upscale follows the display resolution, otherwise the render one.
        target.followsUpscale =
            f.referenceMaxUpscale[0] != 0 &&
            target.refWidth == f.referenceMaxUpscale[0] &&
            target.refHeight == f.referenceMaxUpscale[1];
        if (!target.role.empty() && target.bytesPerTexel != 0 && target.refBytes != 0) {
            f.targets.push_back(target);
        }
        cursor = close + 1;
    }
}

bool consistent(const Plan& plan, std::string& reason) {
    const auto& f = plan.facts;
    if (plan.modules.empty()) {
        reason = "no plan rows parsed";
        return false;
    }
    std::uint32_t classic = 0;
    std::uint32_t neural = 0;
    std::uint32_t unresolved = 0;
    std::uint64_t groups = 0;
    for (const auto& m : plan.modules) {
        if (m.stage == "compute-classic") {
            ++classic;
        } else if (m.stage == "compute-neural") {
            ++neural;
        }
        groups += m.groups.size();
        if (!m.resolved) {
            ++unresolved;
        }
    }
    if (plan.modules.size() != f.computeModules) {
        reason = "module count: tsv=" + std::to_string(plan.modules.size()) +
                 " json=" + std::to_string(f.computeModules);
        return false;
    }
    if (classic != f.classic || neural != f.neural) {
        reason = "stage split: tsv classic=" + std::to_string(classic) +
                 " neural=" + std::to_string(neural) + " json classic=" +
                 std::to_string(f.classic) + " neural=" + std::to_string(f.neural);
        return false;
    }
    if (groups != f.bindingGroups) {
        reason = "binding group total: tsv=" + std::to_string(groups) +
                 " json=" + std::to_string(f.bindingGroups);
        return false;
    }
    if (unresolved != f.unresolved) {
        reason = "unresolved-module count: reader=" + std::to_string(unresolved) +
                 " json=" + std::to_string(f.unresolved) +
                 " (a binding token the reader cannot map, or a table disagreement)";
        return false;
    }
    std::uint64_t targetBytes = 0;
    for (const auto& t : f.targets) {
        targetBytes += t.refBytes;
    }
    if (f.targets.empty() || targetBytes != f.referenceBytesTotal) {
        reason = "intermediate byte total: targets=" + std::to_string(f.targets.size()) +
                 " sum=" + std::to_string(targetBytes) +
                 " json=" + std::to_string(f.referenceBytesTotal);
        return false;
    }
    return true;
}

SizedTargets sizeTargets(const Facts& facts,
                         std::uint32_t renderWidth,
                         std::uint32_t renderHeight,
                         std::uint32_t upscaleWidth,
                         std::uint32_t upscaleHeight) {
    SizedTargets sized;
    // A host that has not revealed its display resolution yet still needs an honest
    // number, so an unknown upscale size falls back to the plan's OWN reference
    // extent (never a guess at a bigger one); the caller logs which happened.
    const std::uint32_t upW = upscaleWidth != 0 ? upscaleWidth
                                               : facts.referenceMaxUpscale[0];
    const std::uint32_t upH = upscaleHeight != 0 ? upscaleHeight
                                                : facts.referenceMaxUpscale[1];
    for (const auto& t : facts.targets) {
        SizedTargets::Entry entry;
        entry.role = t.role;
        if (t.followsUpscale) {
            entry.width = upW != 0 ? upW : t.refWidth;
            entry.height = upH != 0 ? upH : t.refHeight;
        } else {
            entry.width = renderWidth != 0 ? renderWidth : t.refWidth;
            entry.height = renderHeight != 0 ? renderHeight : t.refHeight;
        }
        entry.bytes = static_cast<std::uint64_t>(entry.width) *
                      static_cast<std::uint64_t>(entry.height) *
                      static_cast<std::uint64_t>(t.bytesPerTexel);
        sized.totalBytes += entry.bytes;
        sized.entries.push_back(std::move(entry));
    }
    return sized;
}

}  // namespace ffgplan




