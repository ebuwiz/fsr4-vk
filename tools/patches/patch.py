# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""SPIR-V helper retained for the standalone INT8 patch tools."""

import argparse
import re
from collections import defaultdict

class IR:
    def __init__(self, text):
        self.lines = text.splitlines()
        self.d = {}
        self.location = {}
        self.blocks = {}
        self.serial = 0
        self.globals = []
        self.annotations = []
        self.report = {}
        block = None
        for n, line in enumerate(self.lines):
            m = re.match(r"\s*(%\w+) = (Op\w+)(?: (.*))?$", line)
            if not m:
                if block:
                    self.blocks[block].append(n)
                continue
            ident, op, args = m.groups()
            args = (args or "").split()
            self.d[ident] = (op, args)
            if op == "OpFunction":
                if self.blocks:
                    raise RuntimeError("Requires a single inlined function")
                block = None
            elif op == "OpLabel":
                block = ident
                self.blocks[block] = []
            self.location[ident] = (block, n)
            if block:
                self.blocks[block].append(n)
        if not self.blocks:
            raise RuntimeError("No function blocks")
        self.entry = next(iter(self.blocks))
        self.decorations = defaultdict(list)
        for line in self.lines:
            m = re.match(r"\s*OpDecorate (%\w+) (.*)", line)
            if m:
                self.decorations[m[1]].append(m[2])

    def fresh(self):
        while True:
            self.serial += 1
            ident = f"%redundancy{self.serial}"
            if ident not in self.d:
                self.d[ident] = ("", [])
                return ident

    def type(self, op, args):
        for ident, definition in self.d.items():
            if definition == (op, args):
                return ident
        ident = self.fresh()
        self.d[ident] = op, args
        self.globals.append(f"{ident} = {op} " + " ".join(args))
        return ident

    def constant(self, ty, value):
        args = [ty, str(value)]
        for ident, definition in self.d.items():
            if definition == ("OpConstant", args):
                return ident
        ident = self.fresh()
        self.d[ident] = "OpConstant", args
        self.globals.append(f"{ident} = OpConstant {ty} {value}")
        return ident

    def value(self, ident):
        op, args = self.d.get(ident, ("", []))
        if op == "OpConstant" and self.d[args[0]][0] == "OpTypeInt":
            return int(args[1], 0)
        if op in ("OpCopyObject", "OpBitcast"):
            return self.value(args[1])
        return None

    def pointer_storage(self, ident):
        op, args = self.d.get(ident, ("", []))
        if not args:
            return None
        ty_op, ty_args = self.d.get(args[0], ("", []))
        return ty_args[0] if ty_op == "OpTypePointer" else None

    def finish(self):
        lines = self.lines[:]
        # SPIR-V 1.4+ requires resource globals in the entry-point interface.
        globals_used = [
            i
            for i, (op, args) in self.d.items()
            if op == "OpVariable"
            and args[1] != "Function"
            and self.location.get(i, (None,))[0] is None
        ]
        for n, line in enumerate(lines):
            if "OpEntryPoint " in line:
                missing = [i for i in globals_used if i not in line.split()]
                lines[n] = line + " " + " ".join(missing)
        if self.annotations:
            first_type = next(n for n, l in enumerate(lines) if "= OpType" in l)
            lines[first_type:first_type] = self.annotations
        if self.globals:
            first_function = next(
                n for n, l in enumerate(lines) if "= OpFunction " in l
            )
            lines[first_function:first_function] = self.globals
        return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="The previous patch catalog is not used by FSR 4.1.2.")
    parser.parse_args()
    parser.error("Use tools/setup.py to prepare the FSR 4.1.2 assets")

if __name__ == "__main__":
    main()
