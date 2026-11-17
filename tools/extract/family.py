# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Encode and decode the prepass and postpass endpoint selection keys."""

from sdk import COLORS


def endpoint_key(pass_index, flags=0, color=0, debug=False):
    if pass_index == 0:
        return (flags & 7) | (color << 3) | ((flags & 8) << 2)
    if pass_index == 26:
        return color | (flags & 4) | (int(bool(debug)) << 3)
    raise ValueError("only the prepass and postpass are indexed by an endpoint key")


def decode_key(phase, pass_index, key):
    if phase == "prepass":
        color = (key >> 3) & 3
        flags = (key & 7) | (((key >> 5) & 1) << 3)
        debug = False
    else:
        color, flags, debug = key & 3, key & 4, bool(key & 8)
    if endpoint_key(pass_index, flags, color, debug) != key:
        raise ValueError(f"{phase} index {key} does not decode to "
                         f"{COLORS[color]}/{flags:#04x}/debug={debug}")
    return color, flags, debug
