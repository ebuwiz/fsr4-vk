#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Compare two live captures frame by frame.

Two runs of a game are not the same sequence of frames, so an A/B of the two
descriptor transports cannot assume a run's frame N lines up with the other run's
frame N. Each capture therefore records what a frame *is* - input_fingerprint over
the app's own colour, depth, motion, exposure and mask images plus the runtime's
incoming history, recurrent, reprojected and scratch state - and what its transport
produced - output_fingerprint over postpass_out. This tool pairs frames on the
input side only, checks the two sides really used different transports, then
compares the captured output payloads byte for byte.

Exit codes: 0 every pair identical, 1 some pair differs, 2 nothing could be paired,
which means the runs never saw the same input frame and the comparison proves
nothing rather than that the transports agree.
"""
import argparse
import json
import sys
from pathlib import Path


def frames(root):
    index = {}
    for path in sorted(root.rglob('ab-hashes.json')):
        ab = json.loads(path.read_text())
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        key = (ab['input_fingerprint'],
               manifest['constants'].get('constants_fingerprint'),
               manifest['use_wmma'],
               manifest['run_auto_exposure'],
               manifest['use_staging_output'])
        index.setdefault(key, []).append((path.parent, ab, manifest))
    return index


def payload_files(directory, ab, prefix):
    return [directory / row['path'] for row in ab['files'] if row['path'].startswith(prefix)]


def compare_outputs(frame_a, frame_b, ab_a, ab_b):
    """Byte-compare every postpass_out payload both captures hold."""
    rows_a = {row['path']: row for row in ab_a['files'] if row['path'].startswith('postpass_out/')}
    rows_b = {row['path']: row for row in ab_b['files'] if row['path'].startswith('postpass_out/')}
    missing = set(rows_a) ^ set(rows_b)
    if missing:
        return 'file sets differ: ' + ', '.join(sorted(missing))

    problems = []
    for name, row_a in sorted(rows_a.items()):
        data_a = (frame_a / name).read_bytes()
        data_b = (frame_b / name).read_bytes()
        if data_a == data_b:
            continue
        if len(data_a) != len(data_b):
            problems.append(f'{name} length {len(data_a)} vs {len(data_b)}')
            continue
        offsets = [index for index, (first, second) in enumerate(zip(data_a, data_b))
                   if first != second]
        problems.append(f'{name} differs at {len(offsets)} of {len(data_a)} bytes, '
                        f'first at 0x{offsets[0]:x}')
    return '; '.join(problems) if problems else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_a', type=Path, help='first live capture root')
    parser.add_argument('capture_b', type=Path, help='second live capture root')
    parser.add_argument('--all-pairs', action='store_true',
                        help='print every pair, not only the ones that differ')
    arguments = parser.parse_args()

    for root in (arguments.capture_a, arguments.capture_b):
        if not root.is_dir():
            raise SystemExit(f'not a capture directory: {root}')

    left = frames(arguments.capture_a)
    right = frames(arguments.capture_b)
    shared = [key for key in left if key in right]

    transports_a = {ab['transport'] for entries in left.values() for _, ab, _ in entries}
    transports_b = {ab['transport'] for entries in right.values() for _, ab, _ in entries}

    print(f'captures a={len(left)} b={len(right)} paired={len(shared)} '
          f'transports_a={",".join(sorted(transports_a))} transports_b={",".join(sorted(transports_b))}')

    if not shared:
        print('no pair could be formed: the two runs never received byte-identical '
              'input frames, so this comparison says nothing about the transports')
        return 2

    identical = 0
    differing = 0
    for key in shared:
        # A capture frame can only be recorded once per run, so each side holds at
        # most one entry per key; more than one means the arm fired twice and the
        # pairing is ambiguous, which is worth saying out loud.
        entry_a = left[key][0]
        entry_b = right[key][0]
        frame_a, ab_a, _ = entry_a
        frame_b, ab_b, _ = entry_b

        if ab_a['transport'] == ab_b['transport']:
            print(f'frame={ab_a["frame"]} vs frame={ab_b["frame"]} transport={ab_a["transport"]} '
                  'warning=same transport on both sides')

        # Pixels are judged on the payload bytes, never on the fingerprints: the
        # fingerprints exist to pair frames, and a run whose payload file was short-
        # written would otherwise look identical to the other side. Where the two
        # disagree, that is itself worth reporting, because it means one capture
        # cannot be trusted rather than that the transports differ.
        problem = compare_outputs(frame_a, frame_b, ab_a, ab_b)
        fingerprints_agree = ab_a['output_fingerprint'] == ab_b['output_fingerprint']
        if problem is None and not fingerprints_agree:
            problem = (f'payload bytes match but output fingerprints differ '
                       f'{ab_a["output_fingerprint"]} vs {ab_b["output_fingerprint"]}: '
                       'one capture is stale or short-written')

        if problem is None:
            identical += 1
            if arguments.all_pairs:
                print(f'frame={ab_a["frame"]} vs frame={ab_b["frame"]} input={key[0]} '
                      'output_identical=yes')
        else:
            differing += 1
            print(f'frame={ab_a["frame"]} vs frame={ab_b["frame"]} input={key[0]} '
                  f'output_identical=no {problem}')

    only_a = len([key for key in left if key not in right])
    only_b = len([key for key in right if key not in left])
    print(f'pairs={len(shared)} identical={identical} differing={differing} '
          f'unpaired_a={only_a} unpaired_b={only_b}')

    if differing:
        return 1
    if identical == 0:
        print('no pair carried a postpass_out payload to compare; '
              'run with FSR4_VK_USE_STAGING_OUTPUT=1 to capture the output image')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
