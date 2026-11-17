# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Read the pinned FSR 4.1.2.4555 PE image without loading or executing the DLL.

Every address is derived from the image itself: shader records (0x130 each),
DXBC containers and the fp8 weight banks are located by their structure, and
the semantic meaning of a blob is pinned by its sha256 in
``keys/<version>.json``.  No address from any other SDK build is used.
"""

from pathlib import Path
import hashlib, json, re, shutil, struct, subprocess

SDK_HASH = "aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace"
SDK_VERSION = "4.1.2.4555"
# The asset catalog this SDK build produces is addressed by its own version
# ("fp8/shaders/<ASSET_VERSION>/stock/...", "int8/models/<ASSET_VERSION>/...").
# Every tool and loader path that spells that grammar reads it from here, so a
# new SDK build only has to change SDK_VERSION above.
ASSET_VERSION = ".".join(SDK_VERSION.split(".")[:3])
KEYS_DIR = Path(__file__).resolve().parent / "keys"
RECORD_STRIDE = 0x130
WEIGHT_BANK = 0x20000
TIERS = ("1080", "2160", "4320")
PRESETS = ("native", "quality", "balanced", "performance", "custom", "ultraperf")
COLORS = ("linear", "nonlinear", "srgb", "pq")
SELECTION_FIELDS = [
    (0, 8, 8),
    (8, 0, 4),
    (16, 16, 8),
    (24, 24, 4),
    (28, 64, 4),
    (32, 104, 4),
    (36, 144, 4),
    (40, 184, 4),
    (44, 224, 4),
    (48, 264, 4),
    (56, 32, 8),
    (64, 40, 8),
    (72, 48, 8),
    (80, 56, 8),
    (88, 72, 8),
    (96, 80, 8),
    (104, 88, 8),
    (112, 96, 8),
    (120, 112, 8),
    (128, 120, 8),
    (136, 128, 8),
    (144, 136, 8),
    (152, 152, 8),
    (160, 160, 8),
    (168, 168, 8),
    (176, 176, 8),
    (184, 192, 8),
    (192, 200, 8),
    (200, 208, 8),
    (208, 216, 8),
    (216, 232, 8),
    (224, 240, 8),
    (232, 248, 8),
    (240, 256, 8),
    (248, 272, 8),
    (256, 280, 8),
    (264, 288, 8),
    (272, 296, 8),
]
MODEL_PREFIX = "fsr4_model_"
INSTRUCTION = re.compile(r"^\s*([0-9a-f]+):\s+(.*)$")
CASE_LIMIT = re.compile(r"^cmp\tdword ptr \[[^]]*\], 0x([0-9a-f]+)$")
INDEXED_LOAD = re.compile(r"^mov\t\w+, dword ptr \[\w+ \+ 4\*\w+ \+ (0x[0-9a-f]+)\]$")
RECORD_MULTIPLY = re.compile(r"^imul\t(\w+), \1, 0x130$")
RIP_POINTER = re.compile(r"^lea\t\w+, \[rip [+-] 0x[0-9a-f]+\]\s+# (0x[0-9a-f]+)$")
PRESET_GUARD = re.compile(r"^cmp\tdword ptr \[[^]]*\], 0x5$")
ENDPOINT_KEYS = 64
CASE_BRANCHES = 6


class SDK:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        if self.sha256 != SDK_HASH:
            raise ValueError(
                f"unsupported DLL: expected FSR {SDK_VERSION} ({SDK_HASH[:12]}...), "
                f"got {self.sha256[:12]}..."
            )
        self.version = SDK_VERSION
        pe = struct.unpack_from("<I", self.data, 0x3C)[0]
        count = struct.unpack_from("<H", self.data, pe + 6)[0]
        opt = pe + 24
        size = struct.unpack_from("<H", self.data, pe + 20)[0]
        self.image_base = struct.unpack_from("<Q", self.data, opt + 24)[0]
        self.sections = []
        self.section_names = []
        for i in range(count):
            entry = opt + size + i * 40
            self.section_names.append(self.data[entry : entry + 8].rstrip(b"\0").decode("latin-1"))
            virtual_size, rva, raw_size, raw = struct.unpack_from(
                "<4I", self.data, entry + 8
            )
            self.sections.append((rva, raw, raw_size))
        self._keys = None
        self._image = None
        self._containers = None
        self._record_arrays = None
        self._record_slots = None
        self._instructions = None
        self._dispatch_pairs = None

    def offset(self, rva, size=1):
        for start, raw, length in self.sections:
            if start <= rva and rva + size <= start + length:
                return raw + rva - start
        raise ValueError(f"Unmapped SDK RVA/size: {rva:#x}/{size}")

    def read(self, rva, size):
        offset = self.offset(rva, size)
        return self.data[offset : offset + size]

    def uint(self, rva):
        return struct.unpack("<I", self.read(rva, 4))[0]

    # --- discovery for the 4.1.2 layout: no inherited addresses -------------

    def image(self):
        """PE view that keeps section names, used by the structural discovery."""
        if self._image is None:
            import pe_inventory as inventory
            self._image = inventory.Image(self.path, expect_sha256=None)
        return self._image

    def keys(self):
        """Semantic labels for this exact build, verified against the image."""
        if self._keys is None:
            path = KEYS_DIR / f"{self.version}.json"
            if not path.is_file():
                raise ValueError(f"no semantic key map for SDK {self.version}")
            keys = json.loads(path.read_text())
            if keys.get("source_sha256") != self.sha256:
                raise ValueError(f"{path.name} is pinned to a different DLL")
            self._keys = keys
        return self._keys

    def containers(self):
        if self._containers is None:
            import pe_inventory as inventory
            self._containers = inventory.containers(self.image())
        return self._containers

    def record_arrays(self):
        if self._record_arrays is None:
            import pe_inventory as inventory
            self._record_arrays = inventory.arrays(
                inventory.records(self.image(), self.containers())
            )
        return self._record_arrays

    def named_records(self, entry):
        """Every shader record whose DXBC carries `entry` as its first name."""
        found = []
        for array in self.record_arrays():
            for slot in array["slots"]:
                if slot["name"] == entry:
                    found.append(dict(slot, array=array["index"]))
        return found

    def bank_chain(self, offset, bank):
        """Follow the run of identical `bank`-sized copies at ~0x20000 spacing."""
        blob = self.data[offset : offset + bank]
        chain, current = [offset], offset
        while True:
            for drift in range(0, 0x41, 0x10):
                follower = current + bank + drift
                if follower + bank <= len(self.data) and \
                        self.data[follower : follower + bank] == blob:
                    chain.append(follower)
                    current = follower
                    break
            else:
                return chain, blob

    def weight_chains(self, bank=WEIGHT_BANK):
        """Every run of identical `bank`-sized model banks in the image.

        The model weights sit in `.rdata` as back to back copies at bank spacing
        with a small alignment drift, so a run is recognised by its own content
        repeating and then walked out with `bank_chain`.  One DXIL format the
        upscaler can bind contributes one run, which is how the fp8 and the
        quantized family are told apart without knowing any address.
        """
        runs = []
        for section in self.image().sections:
            if section["name"] not in (".rdata", ".data"):
                continue
            raw, size = section["raw_offset"], section["raw_size"]
            for start in range(raw, raw + size - 2 * bank, 0x10):
                if self.data[start + bank : start + bank + 0x48].find(
                        self.data[start : start + 4]) < 0:
                    continue
                chain, blob = self.bank_chain(start, bank)
                if len(chain) < 3:
                    continue
                if blob.count(blob[:4]) * 4 > len(blob) - 4:
                    continue  # a fill pattern repeated at bank spacing, not a model
                if any(chain[0] > other[0][0] and chain[0] <= other[0][-1]
                       for other in runs):
                    continue  # a copy of a run this walk already covered
                runs.append((chain, blob))
        return sorted(runs, key=lambda run: run[0][0])

    def weight_banks(self, family="fp8", bank=WEIGHT_BANK, overrides=None):
        """The {preset: file offset} model weight banks of one quantisation.

        Both families store one bank shared by the five non Ultra Performance
        presets, repeated once per preset, and one different bank for Ultra
        Performance; the copies sit at bank spacing with a small alignment
        drift, so a candidate position is `chain start + k * bank + drift` for
        a small k and a drift below a page.  Every candidate is hashed, and the
        sha256 pinned in `keys/<version>.json` says which candidates belong to
        this family - never an address from another build.  The five standard
        copies are byte identical, so naming them by ascending address cannot
        change the bytes that get written out.
        """
        name = "weights" if family == "fp8" else "int8_weights"
        pins = overrides or self.keys().get(name) or {}
        wanted = {kind: pins.get(f"{kind}_sha256")
                  for kind in ("standard", "ultraperf")}
        if not all(wanted.values()):
            raise ValueError(
                f'keys/{self.version}.json has no "{name}" weight pins: pin them from '
                "this image with tools/extract/int8.py --dll <dll> --write"
                if family != "fp8" else
                f'keys/{self.version}.json has no "weights" weight pins')
        presets = [preset for preset in PRESETS if preset != "ultraperf"]
        for chain, blob in self.weight_chains(bank):
            if hashlib.sha256(blob).hexdigest() != wanted["standard"]:
                continue
            found = {kind: set() for kind in wanted}
            for step in range(-4, len(chain) + 4):
                for drift in range(-0x40, 0x41, 0x10):
                    offset = chain[0] + step * bank + drift
                    if not 0 <= offset <= len(self.data) - bank:
                        continue
                    digest = hashlib.sha256(self.data[offset : offset + bank]).hexdigest()
                    for kind, pin in wanted.items():
                        if digest == pin:
                            found[kind].add(offset)
            if len(found["standard"]) < len(presets) or len(found["ultraperf"]) != 1:
                continue  # a copy run of one family, not the whole bank set
            banks = dict(zip(presets, sorted(found["standard"])[: len(presets)]))
            banks["ultraperf"] = found["ultraperf"].pop()
            return banks
        raise ValueError(f"no complete {family} weight bank set matches its pinned content")

    def weights(self, bank=WEIGHT_BANK):
        """The fp8 weight banks, found by their copy pattern inside the image.

        The build stores the standard bank five times in a row and a different
        bank immediately before it, which is the Ultra Performance one.  The
        five standard copies are byte identical, so naming them by ascending
        address cannot change the bytes that get written out.
        """
        return self.weight_banks("fp8", bank)

    def record_blob(self, entry, sha256, size=None):
        """One DXBC container, addressed by entry name and pinned content hash."""
        found = [slot for slot in self.named_records(entry) if slot["sha256"] == sha256]
        if len(found) != 1:
            raise ValueError(f"{entry}: {len(found)} records match the pinned hash")
        slot = found[0]
        if size is not None and slot["size"] != size:
            raise ValueError(f"{entry}: container is {slot['size']} bytes, pinned {size}")
        return {"rva": slot["dxbc_rva"], "record_rva": slot["rva"], "size": slot["size"],
                "sha256": slot["sha256"], "data": self.read(slot["dxbc_rva"], slot["size"])}

    def auto_exposure(self):
        """The shared auto-exposure pass, found by DXBC entry name and hash."""
        spec = self.keys()["auto_exposure"]
        return self.record_blob(spec["entry"], spec["dxbc_sha256"], spec.get("size"))

    def rcas(self):
        """The five sharpening passes, addressed by their pinned content hash."""
        spec = self.keys()["rcas"]
        return {mode: self.record_blob(spec["entry"], pin["dxbc_sha256"], pin.get("size"))
                for mode, pin in sorted(spec["modes"].items())}

    # --- the shipping dispatch: (pass, tier, preset) -> record selection -----

    def section_bounds(self, name):
        """The RVA range of one PE section."""
        for section, (rva, _, length) in zip(self.section_names, self.sections):
            if section == name:
                return rva, rva + length
        raise ValueError(f"no {name} section in the image")

    def instructions(self):
        """`.text` as (rva, instruction text) rows, from one llvm-objdump pass."""
        if self._instructions is None:
            tool = shutil.which("llvm-objdump")
            if tool is None:
                raise ValueError("llvm-objdump is needed to read the shader dispatch")
            listing = subprocess.run(
                [tool, "-d", "--no-show-raw-insn", "--x86-asm-syntax=intel",
                 "--section=.text", str(self.path)],
                check=True, capture_output=True, text=True,
            ).stdout
            rows = [(int(match.group(1), 16) - self.image_base, match.group(2))
                    for line in listing.splitlines() if (match := INSTRUCTION.match(line))]
            if len(rows) < 1000:
                raise ValueError("llvm-objdump found no .text instructions to decode")
            self._instructions = rows
        return self._instructions

    def record_slots(self):
        """Every shader record the structural discovery found, keyed by RVA."""
        if self._record_slots is None:
            self._record_slots = {}
            for array in self.record_arrays():
                for slot in array["slots"]:
                    self._record_slots[slot["rva"]] = slot
        return self._record_slots

    def dispatch_pairs(self):
        """Every `record = array + table[key] * 0x130` selection in `.text`.

        The compiler copies the record field by field and reloads both pointers
        for every field, so one branch appears as the same (index table, record
        array) RIP pair repeated back to back; consecutive repeats are one
        branch of the dispatch, not several.
        """
        if self._dispatch_pairs is None:
            rows = self.instructions()
            pairs, previous = [], None
            for index, (rva, text) in enumerate(rows):
                if RECORD_MULTIPLY.match(text) is None:
                    continue
                table = array = None
                for back in range(max(0, index - 8), index):
                    pointer = RIP_POINTER.match(rows[back][1])
                    if pointer:
                        table = int(pointer.group(1), 16) - self.image_base
                for ahead in range(index + 1, min(len(rows), index + 6)):
                    pointer = RIP_POINTER.match(rows[ahead][1])
                    if pointer:
                        array = int(pointer.group(1), 16) - self.image_base
                        break
                if table is not None and array is not None and (table, array) != previous:
                    pairs.append((rva, table, array))
                    previous = (table, array)
            self._dispatch_pairs = pairs
        return self._dispatch_pairs

    def dispatch_switches(self):
        """Jump tables whose cases select shader records.

        The pass dispatch compiles to `cmp <pass>, N` plus a jump table of N+1
        code addresses.  A table only counts once every case in it lands inside
        `.text`, which is how the number of passes and their order are read out
        of the image instead of being written down here.
        """
        rows = self.instructions()
        low, high = self.section_bounds(".text")
        switches = []
        for index, (rva, text) in enumerate(rows):
            limit = CASE_LIMIT.match(text)
            if limit is None:
                continue
            count = int(limit.group(1), 16) + 1
            if not 2 <= count <= 64:
                continue
            table = None
            for ahead in range(index + 1, min(len(rows), index + 12)):
                load = INDEXED_LOAD.match(rows[ahead][1])
                if load:
                    table = int(load.group(1), 16)
                    break
            if table is None:
                continue
            try:
                cases = sorted(self.uint(table + 4 * step) for step in range(count))
            except ValueError:
                continue
            if all(low + 4 <= case < high for case in cases):
                switches.append({"table": table, "cases": cases})
        return switches

    def table_records(self, table, array):
        """The records one (index table, record array) pair is able to select.

        Index tables sit next to each other in `.rdata`, so the table ends where
        its records stop belonging to the pass the branch selects: that leading
        run is the key range the dispatch branch can reach.
        """
        slots = self.record_slots()
        found, entry = {}, None
        try:
            for key in range(ENDPOINT_KEYS):
                index = self.uint(table + 4 * key)
                slot = slots.get(array + index * RECORD_STRIDE)
                if slot is None:
                    break
                if entry is None:
                    entry = slot["name"]
                elif slot["name"] != entry:
                    break
                found[key] = slot
        except ValueError:
            pass
        return found

    def case_selection(self, pairs):
        """Resolve one dispatch case into (entry name, branches), stopping there.

        A case ends where the records it selects stop sharing one DXBC entry
        name, which also bounds the last case of a switch.
        """
        chosen, entry = [], None
        for site, table, array in pairs:
            found = self.table_records(table, array)
            names = {slot["name"] for slot in found.values()}
            if not found or len(names) != 1:
                break
            name = names.pop()
            if entry is None:
                entry = name
            elif name != entry:
                break
            chosen.append((site, table, array, found))
        return entry, chosen

    def dxil_version(self, rva, size):
        """The DXIL program version a DXBC container was compiled for.

        The per-shader DXC build number varies inside one build, so the program
        version record is what identifies the compiler format a dispatcher uses.
        """
        if self.read(rva, 4) != b"DXBC":
            raise ValueError(f"{rva:#x} is not a DXBC container")
        table = self.read(rva + 0x20, 4 * self.uint(rva + 0x1C))
        for index in range(len(table) // 4):
            offset = int.from_bytes(table[index * 4 : index * 4 + 4], "little")
            if self.read(rva + offset, 4) == b"DXIL":
                return self.read(rva + offset + 8, 4)
        raise ValueError(f"{rva:#x} has no DXIL region")

    def shader_format(self):
        """The DXIL format of the pinned non-model shaders in this same image."""
        blob = self.auto_exposure()
        return self.dxil_version(blob["rva"], blob["size"])

    def pinned_record(self, record_rva, sha256, size=None, entry=None):
        """One shader record by address, its container verified by content hash.

        Record arrays hold several variants of the same entry and different
        records can point at byte identical containers, so an entry name plus a
        hash alone is not enough to name one record: the dispatch decides which
        record is the shipping one, and the pinned hash confirms what it holds.
        """
        record = self.read(record_rva, RECORD_STRIDE)
        rva, data = self.shader(record)
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            raise ValueError(f"record {record_rva:#x} holds {digest}, pinned {sha256}")
        if size is not None and len(data) != size:
            raise ValueError(f"record {record_rva:#x} holds {len(data)} bytes, pinned {size}")
        slot = self.record_slots().get(record_rva)
        if slot is not None and entry is not None and slot["name"] != entry:
            raise ValueError(f"record {record_rva:#x} is {slot['name']}, expected {entry}")
        return {"rva": rva, "record_rva": record_rva, "size": len(data),
                "sha256": digest, "data": data,
                "entry": entry if slot is None else slot["name"]}

    def shader(self, record):
        size = struct.unpack_from("<I", record)[0]
        pointer = struct.unpack_from("<Q", record, 8)[0]
        if not 28 <= size <= 64 * 1024 * 1024:
            raise ValueError("Invalid selected shader size")
        rva = pointer - self.image_base
        data = self.read(rva, size)
        if data[:4] != b"DXBC" or struct.unpack_from("<I", data, 24)[0] != size:
            raise ValueError("Invalid selected DXBC container")
        return rva, data

    def selection(self, record):
        result = bytearray(0x118)
        for target, source, size in SELECTION_FIELDS:
            result[target : target + size] = record[source : source + size]
        return bytes(result)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def extract_rcas(sdk, output):
    """The five sharpening passes, found by DXBC entry name and content hash."""
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh RCAS extraction directory")
    output.mkdir(parents=True)
    records = {}
    for name, shader in sdk.rcas().items():
        if shader["data"][:4] != b"DXBC":
            raise ValueError("invalid RCAS container for " + name)
        write(output / (name + ".dxbc"), shader["data"])
        records[name] = {"rva": shader["rva"], "record_rva": shader["record_rva"],
                         "size": shader["size"], "sha256": shader["sha256"]}
    (output / "manifest.json").write_text(json.dumps({
        "sdk_version": sdk.version, "source_sha256": sdk.sha256, "shaders": records,
    }, indent=2) + "\n")
    return records


def extract_weights(sdk, output):
    """The six fp8 preset weight banks (128 KiB each)."""
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh weight extraction directory")
    output.mkdir(parents=True)
    manifest = {}
    for preset, offset in sdk.weights().items():
        data = sdk.data[offset:offset + WEIGHT_BANK]
        write(output / f"weights_{preset}.bin", data)
        manifest[preset] = {
            "file": f"weights_{preset}.bin", "offset_in_dll": offset,
            "rva": sdk.image().rva(offset), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps({
        "sdk_version": sdk.version, "source_sha256": sdk.sha256,
        "bank_size": WEIGHT_BANK, "weights": manifest,
    }, indent=2) + "\n")
    return manifest


def extract_auto_exposure(sdk, output):
    """The shared auto-exposure pass (single pass downsample)."""
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh auto-exposure output directory")
    output.mkdir(parents=True)
    shader = sdk.auto_exposure()
    write(output / "auto-exposure.dxbc", shader["data"])
    entry = {"file": "auto-exposure.dxbc", "rva": shader["rva"], "size": shader["size"],
             "sha256": shader["sha256"]}
    (output / "manifest.json").write_text(json.dumps({
        "sdk_version": sdk.version, "source_sha256": sdk.sha256, "auto_exposure": entry,
    }, indent=2) + "\n")
    return entry
