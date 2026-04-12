"""
stage1_exact.py — Exact statevector simulation for P1 and P2
=============================================================

P1 (4 qubits): product state — solved analytically in milliseconds.
P2 (28 qubits): exact statevector with ~2.1 GB RAM.

Both use a reshape-based gate application strategy: instead of building
explicit index arrays, we reshape the statevector into a tensor, apply
the gate on the target axes, and reshape back. This avoids large
intermediate allocations.

Usage:
    python stage1_exact.py --qasm P1_little_peak.qasm
    python stage1_exact.py --qasm P2_swift_rise.qasm
"""

import argparse
import re
import sys
import time
import numpy as np
from core_mps import rz_mat, ry_mat, u3_mat, parse_angle, X_MAT, SX_MAT

DTYPE = np.complex64


# ─── Statevector Gate Application ────────────────────────────────────────────

def apply_1q(sv: np.ndarray, U: np.ndarray, qubit: int, n: int) -> None:
    """
    Apply 2×2 unitary U to `qubit` in the n-qubit statevector.

    Reshapes sv to (2^(n-qubit-1), 2, 2^qubit), applies U on the
    middle axis (qubit axis), then restores shape. No copies except
    the small chunk that's being modified.
    """
    A = 1 << (n - qubit - 1)
    B = 1 << qubit
    sv3 = sv.reshape(A, 2, B)
    c0 = sv3[:, 0, :].copy()
    c1 = sv3[:, 1, :].copy()
    sv3[:, 0, :] = U[0, 0] * c0 + U[0, 1] * c1
    sv3[:, 1, :] = U[1, 0] * c0 + U[1, 1] * c1


def apply_cz(sv: np.ndarray, q0: int, q1: int, n: int) -> None:
    """
    CZ gate: flip sign of amplitude where both q0=1 and q1=1.

    Reshapes sv to expose both qubit axes simultaneously:
    (2^(n-q1-1), 2, 2^(q1-q0-1), 2, 2^q0)
    and negates the [1, :, 1, :] slice.
    """
    if q0 > q1:
        q0, q1 = q1, q0
    sv.reshape(
        1 << (n - q1 - 1), 2,
        1 << (q1 - q0 - 1), 2,
        1 << q0
    )[:, 1, :, 1, :] *= -1


# ─── QASM Parser ──────────────────────────────────────────────────────────────

def parse_and_simulate(qasm_path: str) -> str:
    """
    Parse a QASM file and simulate it as an exact statevector.
    Returns the peak bitstring in Qiskit convention (q[n-1]...q[0]).
    """
    qasm = open(qasm_path).read()
    n = int(re.search(r'qreg\s+\w+\[(\d+)\]', qasm).group(1))

    ram_gb = (1 << n) * 8 / 1e9
    print(f"  Qubits: {n}  |  States: {2**n:,}  |  RAM: {ram_gb:.2f} GB")

    if n > 29:
        print(f"  WARNING: {n} qubits requires {ram_gb:.1f} GB RAM — may OOM.")

    sv = np.zeros(1 << n, dtype=DTYPE)
    sv[0] = 1.0

    gate_count = 0
    t0 = time.time()

    for line in qasm.strip().splitlines():
        line = line.strip().rstrip(';')
        if not line or any(line.startswith(k) for k in
                           ['OPENQASM', 'include', 'qreg', '//']):
            continue

        # CZ
        m = re.match(r'^cz\s+\w+\[(\d+)\],\w+\[(\d+)\]$', line)
        if m:
            apply_cz(sv, int(m.group(1)), int(m.group(2)), n)
            gate_count += 1
            continue

        # U3
        m = re.match(r'^u3\(([^)]+)\)\s+\w+\[(\d+)\]$', line)
        if m:
            params = [parse_angle(p) for p in m.group(1).split(',')]
            apply_1q(sv, u3_mat(*params), int(m.group(2)), n)
            gate_count += 1
            continue

        # RZ
        m = re.match(r'^rz\(([^)]+)\)\s+\w+\[(\d+)\]$', line)
        if m:
            apply_1q(sv, rz_mat(parse_angle(m.group(1))), int(m.group(2)), n)
            gate_count += 1
            continue

        # RY
        m = re.match(r'^ry\(([^)]+)\)\s+\w+\[(\d+)\]$', line)
        if m:
            apply_1q(sv, ry_mat(parse_angle(m.group(1))), int(m.group(2)), n)
            gate_count += 1
            continue

        # SX
        m = re.match(r'^sx\s+\w+\[(\d+)\]$', line)
        if m:
            apply_1q(sv, SX_MAT, int(m.group(1)), n)
            gate_count += 1
            continue

        # X
        m = re.match(r'^x\s+\w+\[(\d+)\]$', line)
        if m:
            apply_1q(sv, X_MAT, int(m.group(1)), n)
            gate_count += 1
            continue

    elapsed = time.time() - t0
    print(f"  {gate_count} gates applied in {elapsed:.2f}s")

    # Compute probabilities in chunks to avoid large intermediate arrays
    S = 1 << n
    C = 1 << 22  # 4M elements at a time
    probs = np.empty(S, dtype=np.float32)
    for i in range(0, S, C):
        j = min(i + C, S)
        chunk = sv[i:j]
        probs[i:j] = chunk.real**2 + chunk.imag**2

    norm = float(probs.sum())
    print(f"  State norm: {norm:.8f}")

    # Top 5 bitstrings
    top5_idx = np.argpartition(probs, -5)[-5:]
    top5_idx = top5_idx[np.argsort(probs[top5_idx])[::-1]]

    print(f"\n  Top 5 bitstrings (Qiskit format q[n-1]...q[0]):")
    for rank, idx in enumerate(top5_idx):
        bs = format(int(idx), f'0{n}b')
        marker = '  ← PEAK' if rank == 0 else ''
        print(f"    {bs}  prob={probs[idx]:.8f}{marker}")

    peak_idx = int(top5_idx[0])
    peak_bs = format(peak_idx, f'0{n}b')  # Qiskit convention: MSB = q[n-1]

    # Also show ratio above uniform
    uniform = 1.0 / (1 << n)
    ratio = probs[peak_idx] / uniform
    print(f"\n  Peak/uniform ratio: {ratio:.2e}×")
    print(f"\n  PEAK:    {peak_bs}")
    print(f"  REVERSE: {peak_bs[::-1]}")

    return peak_bs


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Exact statevector simulation (P1, P2)')
    parser.add_argument('--qasm', required=True, help='Path to .qasm file')
    args = parser.parse_args()

    print('=' * 60)
    print('  Stage 1 — Exact Statevector Simulation')
    print('=' * 60)

    parse_and_simulate(args.qasm)


if __name__ == '__main__':
    main()
