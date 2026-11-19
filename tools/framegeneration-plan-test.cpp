// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// GPU-free consistency test for the Frame Generation pass-plan substrate (roadmap Phase
// 0). It re-parses the pack's own `pack/pipelines.tsv` layout column with the SAME grammar
// tools/framegeneration-exec.cpp uses to build real descriptor-set layouts, then asserts that
// independent reconstruction agrees, module-for-module and count-for-count, with the plan that
// `tools/extract/frame-generation/plan.py` wrote to `pack/fg-plan.tsv`. Two parsers (Python
// plan.py, this C++ one) agreeing on the de-duplicated binding census — and on 1208 total binding
// groups across the 474 compute modules — is what makes the plan trustworthy as the substrate the
// provider's Prepare/Generate recording (Phases 1-2) and GPU-memory-usage report (Phase 3) build
// on. It asserts STRUCTURE, not pixel parity with AMD's DLL: the plan's ORDER is a grouped
// scaffold (classic-first, mlfi tier-grouped, pass-ascending), explicitly NOT claimed to match
// AMD's execution order. No device, no assets beyond the two TSVs. Exits 0 OK, 1 regression,
// 77 skip (FG pack/plan not generated on this checkout).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <regex>

static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (true) {
        const auto pos = s.find(d, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

// Canonical de-duplicated binding-group string for a pipelines.tsv `layout` value.
// Grammar mirrors tools/framegeneration-exec.cpp: tokens ';'-separated, each
// `set:binding:KIND[:a]`; kinds in {SB,UB,CIS,SI,SMP}; a short/'U' token = unresolved;
// same (set,binding) with two kinds = cross-kind (unified). Groups are sorted by
// (set, binding) so this parser and plan.py compare on a stable order.
static const std::set<std::string> kKinds = {"SB", "UB", "CIS", "SI", "SMP"};
static bool unresolvedToken(const std::string& tok) {
    const auto f = split(tok, ':');
    return f.size() < 3 || kKinds.find(f[2]) == kKinds.end();
}
static std::string canonicalGroups(const std::string& layout, bool* unresolved) {
    // (set,binding) -> (first-seen kind, OR-merged arrayed flag), identical to the
    // executor's de-dup: a slot claimed by a second kind is cross-kind (unified), a
    // repeated same kind collapses but keeps `arrayed` if any view into it is arrayed.
    std::map<std::pair<int, int>, std::pair<std::string, bool>> g;
    for (const auto& tok : split(layout, ';')) {
        if (tok.empty()) {
            continue;
        }
        if (unresolvedToken(tok)) {
            *unresolved = true;
            continue;
        }
        const auto f = split(tok, ':');
        const bool arr = f.size() >= 4 && f[3] == "a";
        const auto key = std::make_pair(std::atoi(f[0].c_str()), std::atoi(f[1].c_str()));
        const auto it = g.find(key);
        if (it == g.end()) {
            g.emplace(key, std::make_pair(f[2], arr));
        } else {
            it->second.second = it->second.second || arr;
        }
    }
    std::string cs;
    for (const auto& kv : g) {
        if (!cs.empty()) {
            cs += ';';
        }
        cs += std::to_string(kv.first.first) + ':' + std::to_string(kv.first.second) + ':' + kv.second.first;
        if (kv.second.second) {
            cs += ":a";
        }
    }
    return cs;
}

// Run an external command and capture stdout (used to invoke the SAME spirv-dis the
// SPIR-V validator gate uses). Returns "" if the process cannot be spawned.
static std::string runCapture(const std::string& cmd) {
    std::string out;
    auto* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return out;
    }
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        out += buf;
    }
    pclose(pipe);
    return out;
}

// Locate spirv-dis: FSR4_SPIRV_DIS override, then PATH, then the vendored build path. Empty
// means "no SPIR-V toolchain here" -> the cooperative cross-check self-skips (it must NOT fail
// a host that lacks spirv-dis, same tolerance as the descriptor-buffer/plan-drift on-device gate).
static std::string findSpirvDis() {
    if (const char* e = std::getenv("FSR4_SPIRV_DIS"); e && *e) {
        return e;
    }
    const auto which = runCapture("command -v spirv-dis 2>/dev/null");
    std::string s = which;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    if (!s.empty()) {
        return s;
    }
    std::ifstream direct("/usr/bin/spirv-dis");
    if (direct) {
        return "/usr/bin/spirv-dis";
    }
    std::ifstream vendored("build/tools/SPIRV-Tools/bin/spirv-dis");
    if (vendored) {
        return "build/tools/SPIRV-Tools/bin/spirv-dis";
    }
    return "";
}

// Canonical sorted DISTINCT CooperativeMatrixKHR operand shapes 'comp:scope:rows:cols:K' for one
// module, captured from its SPIR-V disassembly with a C++-native regex -- an INDEPENDENT parser
// from plan.py's Python regex. Both must agree module-for-module on the recovered CsTensorSizes.
static std::vector<std::string> splitWs(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char c : s) {
        if (c == ' ' || c == '\t' || c == '\r') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

// Canonical sorted DISTINCT CooperativeMatrixKHR operand shapes 'comp:scope:rows:cols:K' for one
// module, captured from its SPIR-V disassembly with a C++-native line/token parse -- an INDEPENDENT
// parser from plan.py's Python regex. Both must agree module-for-module on the recovered CsTensorSizes.
static std::string coopSet(const std::string& asmText, std::size_t* shapeCount, bool* any) {
    std::set<std::string> shapes;
    const auto strip = [](const std::string& t) -> std::string {
        if (t.rfind("%uint_", 0) == 0) {
            return t.substr(6);
        }
        if (!t.empty() && t[0] == '%') {
            return t.substr(1);
        }
        return t;
    };
    for (const auto& raw : split(asmText, '\n')) {
        const auto pos = raw.find("OpTypeCooperativeMatrixKHR");
        if (pos == std::string::npos) {
            continue;
        }
        // Layout: [OpTypeCooperativeMatrixKHR, %comp, %uint_<scope>, %uint_<rows>, %uint_<cols>, %uint_<K>]
        const auto toks = splitWs(raw.substr(pos));
        if (toks.size() < 6 || toks[1].empty() || toks[1][0] != '%') {
            continue;
        }
        bool ints = true;
        for (std::size_t i = 2; i <= 5; ++i) {
            if (toks[i].rfind("%uint_", 0) != 0) {
                ints = false;
            }
        }
        if (!ints) {
            continue;
        }
        shapes.insert(strip(toks[1]) + ":" + strip(toks[2]) + ":" + strip(toks[3]) + ":" +
                      strip(toks[4]) + ":" + strip(toks[5]));
    }
    *any = !shapes.empty();
    *shapeCount = shapes.size();
    std::string joined;
    for (const auto& sh : shapes) {
        if (!joined.empty()) {
            joined += ";";
        }
        joined += sh;
    }
    return joined;
}

static bool isMlfi(const std::string& entry) {
    // mlfi_<tier>_pass<digits>, tier nonempty, digits nonempty, nothing after the digits.
    const std::string pre = "mlfi_";
    if (entry.rfind(pre, 0) != 0) {
        return false;
    }
    const auto pass = entry.find("_pass", pre.size());
    if (pass == std::string::npos || pass <= pre.size()) {
        return false;  // empty tier
    }
    bool sawDigit = false;
    for (std::size_t i = pass + 5; i < entry.size(); ++i) {
        if (entry[i] < '0' || entry[i] > '9') {
            return false;
        }
        sawDigit = true;
    }
    return sawDigit;
}

int main() {
    const char* packEnv = std::getenv("FFG4_FG_PACK");
    std::string pack = (packEnv && *packEnv) ? packEnv : "assets/frame-generation/pack";
    std::ifstream pip(pack + "/pipelines.tsv");
    std::ifstream plan(pack + "/fg-plan.tsv");
    if (!pip || !plan) {
        std::printf("framegeneration-plan-test: SKIP (FG pack/plan not present under %s)\n",
                    pack.c_str());
        return 77;
    }

    // Reconstruct the executor-grammar census from pipelines.tsv (independent parser).
    struct Row { std::string groups, stage, entry; };
    std::map<std::string, Row> fromPack;
    std::string line;
    std::getline(pip, line);  // header
    while (std::getline(pip, line)) {
        const auto c = split(line, '\t');
        if (c.size() < 8) {
            continue;
        }
        if (c[1] != "compute-classic" && c[1] != "compute-neural") {
            continue;  // 12 graphics/present/PSV modules are not compute-dispatchable
        }
        bool unresolved = false;
        const auto groups = canonicalGroups(c[6], &unresolved);
        if (unresolved) {
            fromPack[c[0]] = Row{"<UNRESOLVED>", c[1], c[2]};
        } else {
            fromPack[c[0]] = Row{groups, c[1], c[2]};
        }
    }

    // Read plan.py's plan.
    struct P { std::string groups, stage, tier, resolved; long pass = -1; long order = -1;
               long binding_count = -1; std::string coop; };
    std::map<std::string, P> fromPlan;
    std::getline(plan, line);  // header
    while (std::getline(plan, line)) {
        const auto c = split(line, '\t');
        if (c.size() < 11) {
            continue;
        }
        P p;
        p.stage = c[1];
        p.tier = c[3];
        p.pass = c[4].empty() ? -1 : std::atol(c[4].c_str());
        p.order = std::atol(c[5].c_str());
        p.resolved = c[7];
        p.binding_count = std::atol(c[8].c_str());
        p.groups = c[9];
        p.coop = (c.size() >= 12) ? c[11] : std::string();
        fromPlan[c[0]] = p;
    }

    int failures = 0;
    auto fail = [&](const std::string& msg) {
        std::printf("FAIL %s\n", msg.c_str());
        ++failures;
    };

    // 1. Coverage: both maps hold the same compute module set, sized 474.
    if (fromPack.size() != 474) {
        fail("pack compute modules != 474 (got " + std::to_string(fromPack.size()) + ")");
    }
    if (fromPlan.size() != 474) {
        fail("plan modules != 474 (got " + std::to_string(fromPlan.size()) + ")");
    }
    if (fromPack.size() == fromPlan.size()) {
        for (const auto& kv : fromPack) {
            if (fromPlan.find(kv.first) == fromPlan.end()) {
                fail("module in pack not in plan: " + kv.first);
                break;
            }
        }
        for (const auto& kv : fromPlan) {
            if (fromPack.find(kv.first) == fromPack.end()) {
                fail("module in plan not in pack: " + kv.first);
                break;
            }
        }
    }

    // 2. Per-module agreement: the two independent parsers produce identical binding groups,
    //    plan's binding_count matches the group count, and total groups == 1208.
    std::size_t totalGroups = 0;
    std::size_t neural = 0;
    std::size_t classic = 0;
    for (const auto& kv : fromPack) {
        const auto it = fromPlan.find(kv.first);
        if (it == fromPlan.end()) {
            continue;
        }
        if (kv.second.groups != it->second.groups) {
            fail("binding-group drift for " + kv.first +
                 " (executor='" + kv.second.groups + "' plan='" + it->second.groups + "')");
        }
        if (it->second.groups != "<UNRESOLVED>") {
            const auto parts = split(it->second.groups, ';');
            std::size_t n = 0;
            for (const auto& s : parts) {
                if (!s.empty()) {
                    ++n;
                }
            }
            totalGroups += n;
            if (n != static_cast<std::size_t>(it->second.binding_count)) {
                fail("binding_count column != group count for " + kv.first);
            }
        }
        if (kv.second.stage == "compute-neural") {
            ++neural;
            if (!isMlfi(kv.second.entry)) {
                fail("neural entry not mlfi_<tier>_pass<N>: " + kv.second.entry);
            }
        } else {
            ++classic;
        }
    }
    if (neural != 382 || classic != 92) {
        fail("stage split != neural 382 / classic 92 (got " + std::to_string(neural) +
             " / " + std::to_string(classic) + ")");
    }
    if (totalGroups != 1208) {
        fail("distinct binding groups across compute != 1208 (got " +
             std::to_string(totalGroups) + ")");
    }

    // 3. Order is a bijection 0..473; classic precedes neural; mlfi tiers grouped,
    //    pass-ascending within a tier (a deterministic scaffold, not an AMD-order claim).
    std::vector<long> orders;
    for (const auto& kv : fromPlan) {
        orders.push_back(kv.second.order);
    }
    std::sort(orders.begin(), orders.end());
    for (std::size_t i = 0; i < orders.size(); ++i) {
        if (orders[i] != static_cast<long>(i)) {
            fail("plan order column is not a bijection over 0.." +
                 std::to_string(orders.size() - 1));
            break;
        }
    }
    long maxClassic = -1;
    long minNeural = 100000;
    std::vector<std::pair<long, std::pair<std::string, long>>> neuralByOrder;
    for (const auto& kv : fromPlan) {
        if (kv.second.tier == "classic") {
            maxClassic = std::max(maxClassic, kv.second.order);
        } else {
            minNeural = std::min(minNeural, kv.second.order);
            neuralByOrder.emplace_back(
                kv.second.order, std::make_pair(kv.second.tier, kv.second.pass));
        }
    }
    if (!(maxClassic < minNeural)) {
        fail("classic block does not strictly precede the mlfi block in the plan order");
    }
    std::sort(neuralByOrder.begin(), neuralByOrder.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    {
        std::string seenTier;
        std::set<std::string> finishedTiers;
        long lastPass = -1;
        for (const auto& e : neuralByOrder) {
            const std::string& tier = e.second.first;
            const long pass = e.second.second;
            if (tier != seenTier) {
                if (finishedTiers.find(tier) != finishedTiers.end()) {
                    fail("tier '" + tier + "' recurs non-contiguously in the plan order");
                    break;
                }
                if (!seenTier.empty()) {
                    finishedTiers.insert(seenTier);
                }
                seenTier = tier;
                lastPass = -1;
            }
            if (pass < lastPass) {
                fail("mlfi pass indices not nondecreasing within tier '" + tier + "'");
                break;
            }
            lastPass = pass;
        }
    }

    // 4. Every compute module resolved cleanly (FG manifest: no unified, no unresolved).
    for (const auto& kv : fromPlan) {
        if (kv.second.resolved != "1") {
            fail("plan module not resolved: " + kv.first);
            break;
        }
    }

    // 5. CooperativeMatrix operand shapes (the DLL's "CsTensorSizes") re-derived INDEPENDENTLY
    //    from the shipped SPIR-V via spirv-dis, cross-checked against plan.py module-for-module,
    //    then pinned to the measured FG 4.0.2 census. Grounds roadmap blocker #1 (CsTensorSizes) as a
    //    machine-checked fact; it does NOT enable ffxDispatch (graph deps/order, intermediate
    //    allocation and pacing remain unreconstructed, so the provider still refuses dispatch).
    const auto dis = findSpirvDis();
    if (!dis.empty()) {
        std::size_t coopUsing = 0;
        std::size_t coopClassic = 0;
        std::map<std::string, std::size_t> shapeModules;  // shape -> #modules declaring it
        std::map<std::string, std::size_t> tierUsing;      // neural tier -> #modules using coop
        std::map<std::string, std::vector<std::string>> tierGeom;
        std::size_t disagree = 0;
        for (const auto& kv : fromPack) {
            const auto modPath = pack + "/modules/" + kv.first;
            const auto asmText = runCapture("\'" + dis + "\' \'" + modPath + "\'");
            std::size_t nshapes = 0;
            bool any = false;
            const auto derived = coopSet(asmText, &nshapes, &any);
            const auto it = fromPlan.find(kv.first);
            const std::string planCoop =
                (it != fromPlan.end()) ? it->second.coop : std::string("<absent>");
            if (planCoop != derived) {
                ++disagree;
                if (disagree <= 3) {
                    fail("cooperative-shape drift for " + kv.first + " (C++='" + derived +
                         "' plan.py='" + planCoop + "')");
                }
            }
            if (any) {
                ++coopUsing;
                for (const auto& sh : split(derived, ';')) {
                    if (!sh.empty()) {
                        ++shapeModules[sh];
                    }
                }
                const std::string tier = (it != fromPlan.end()) ? it->second.tier : "";
                if (tier == "classic") {
                    ++coopClassic;
                } else {
                    ++tierUsing[tier];
                    tierGeom[tier].push_back(derived);
                }
            }
        }
        if (disagree > 3) {
            fail("cooperative-shape drift on more than 3 modules (see above)");
        }
        if (coopUsing != 156) {
            fail("cooperative-matrix modules != 156 (got " + std::to_string(coopUsing) + ")");
        }
        if (coopClassic != 0) {
            fail("classic modules must use no CooperativeMatrixKHR (got " +
                 std::to_string(coopClassic) + ")");
        }
        // Pinned CsTensorSizes: the exact (shape -> #modules) census the DLL's ML graph runs on.
        const std::map<std::string, std::size_t> expectedShapes = {
            {"float:3:16:16:2", 156},  {"half:3:16:16:0", 156},  {"half:3:16:16:1", 156},
            {"half:3:16:16:2", 142},   {"uchar:3:16:16:1", 32},  {"uchar:3:16:16:0", 30},
        };
        if (shapeModules != expectedShapes) {
            fail("CsTensorSizes census differs from the pinned FG 4.0.2 set (shape or count drift)");
        }
        if (tierUsing["4k"] != 78 || tierUsing["8k"] != 78) {
            fail("cooperative neural tiers != 4k=78 / 8k=78 (got 4k=" +
                 std::to_string(tierUsing["4k"]) + " 8k=" + std::to_string(tierUsing["8k"]) + ")");
        }
        // 4k and 8k share ONE model, so their cooperative GEOMETRY multisets must be identical.
        std::sort(tierGeom["4k"].begin(), tierGeom["4k"].end());
        std::sort(tierGeom["8k"].begin(), tierGeom["8k"].end());
        if (tierGeom["4k"] != tierGeom["8k"]) {
            fail("4k and 8k ML chains must share identical cooperative geometry (shared model)");
        }
    } else {
        std::printf("[coop] SKIP: spirv-dis not found; cooperative CsTensorSizes cross-check not run\n");
    }

    if (failures == 0) {
        std::printf("framegeneration-plan consistency OK: 474 compute modules "
                    "(382 mlfi + 92 classic), 1208 binding groups, order scaffold valid, "
                    "156 CooperativeMatrix modules (6 pinned CsTensorSizes, 4k==8k geometry)\n");
        return 0;
    }
    std::printf("framegeneration-plan consistency FAILED: %d\n", failures);
    return 1;
}


