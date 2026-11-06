#!/usr/bin/env python3
"""Summarize a headless run CSV written by the raylib shim.

usage: summarize.py run.csv [more.csv ...]

Reports, per CSV: how far the organism was from the mouse, how often it was
pinned against a wall, what the motor signals looked like, and how active the
random reservoir was.
"""
import csv
import math
import sys


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k])
    dist = [math.hypot(f(r, "mouse_x") - f(r, "org_x"), f(r, "mouse_y") - f(r, "org_y")) for r in rows]
    wall = sum(1 for r in rows if f(r, "org_x") in (0, 800) or f(r, "org_y") in (0, 600))
    motors = [f(r, k) for r in rows for k in ("left", "right", "up", "down")]
    integer_motor = sum(1 for m in motors if m == int(m)) / len(motors)
    steady = rows[len(rows) // 10:]  # skip the start-up transient
    active = sum(f(r, "active_frac") for r in steady) / len(steady)
    fired = sum(f(r, "fired_frac") for r in steady) / len(steady)

    # Distance to the target at the end of each stretch where the mouse stayed put.
    segments, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or (rows[i]["mouse_x"], rows[i]["mouse_y"]) != (rows[start]["mouse_x"], rows[start]["mouse_y"]):
            reach = next((j - start for j in range(start, i) if dist[j] <= 10), None)
            segments.append((int(rows[start]["frame"]), int(rows[i - 1]["frame"]), dist[i - 1], reach))
            start = i

    print(f"== {path}  ({len(rows)} frames)")
    print(f"   mean distance to mouse: {sum(dist) / len(dist):7.1f} px   final: {dist[-1]:6.1f} px")
    print(f"   frames pinned to a wall: {wall} ({100 * wall / len(rows):.0f}%)")
    print(f"   motor-group sums that are whole numbers: {100 * integer_motor:.1f}%")
    print(f"   reservoir (after warm-up): {100 * active:.1f}% of neurons active, {100 * fired:.1f}% fire per frame")
    print("   per mouse position (frames: distance at the end, frames until within 10 px):")
    for a, b, d, reach in segments:
        print(f"     {a:5d}-{b:<5d} end {d:6.1f} px   reached: {'never' if reach is None else f'after {reach} frames'}")


for p in sys.argv[1:]:
    summarize(p)
