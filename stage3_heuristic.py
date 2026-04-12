"""
stage3_heuristic.py — Approximate MPS for all-to-all circuits
=============================================================

Handles:
  P5 — All-to-All 44q, 1892 CZ  (χ=32, LR=RL confirmed)
  P6 — All-to-All 62q, 3494 CZ  (χ=24, beats χ=16 by +91 nats)
  P9 — All-to-All 56q, 1917 RZZ (χ=24, HQAP — hardest problem)

These circuits have nearly full connectivity, making exact MPS representation
exponentially expensive. We use MPS with moderate χ and rely on:
  1. Multi-χ cross-validation: run at several χ values
  2. Bidirectional beam search: LR vs RL agreement as self-check
  3. Perturbation test: confirm local maximum under best MPS
  4. Cross-χ evaluation: score all candidates under best available MPS

IMPORTANT WARNING FOR P9:
  Do NOT use Ising Simulated Annealing for P9, even though all 1917 RZZ
  gates have the same angle (-π/2). SA ignores the 3890 single-qubit U
  gates that determine the actual peak. SA returns '111...1' — wrong!
  Always simulate the full quantum circuit.

Bit ordering: output in MPS site order (q[0]...q[n-1]).
The challenge accepts both this and its reverse as correct.

Usage:
    python stage3_heuristic.py --qasm P5_granite_summit.qasm
    python stage3_heuristic.py --qasm P6_titan_pinnacle.qasm
    python stage3_heuristic.py --qasm P9_hqap_1917.qasm
"""

import argparse
import os
import re
import sys
import time
import numpy as np
from core_mps import (
    mps_init, mps_apply_1q, mps_apply_nn, mps_apply_lr_gate,
    bidirectional_decode, perturbation_test, mps_logprob,
    max_bond_dim, bond_profile, CZ4, rzz_mat, u3_mat, parse_angle, DTYPE
)

# Validated χ configurations per problem
PROBLEM_CONFIGS = {
    'P5': {'chi_values': [24, 32], 'circuit_type': 'cz',
           'note': 'All-to-all 44q — χ=32 gives LR=RL and local max'},
    'P6': {'chi_values': [16, 24], 'circuit_type': 'cz',
           'note': 'All-to-all 62q — χ=24 beats χ=16 by +91.6 nats'},
    'P9': {'chi_values': [16, 24], 'circuit_type': 'rzz',
           'note': 'HQAP 56q — hardest, do NOT use Ising SA'},
}


# ─── QASM Parsers ─────────────────────────────────────────────────────────────

def parse_cz_u3_alltoall(qasm_path: str):
    """
    Parser for all-to-all CZ + U3 circuits (P5, P6).
    Handles non-standard register names (P5 uses 'q83' instead of 'q').
    """
    qasm = open(qasm_path).read()
    n = int(re.search(r'qreg\s+\w+\[(\d+)\]', qasm).group(1))
    reg = re.search(r'qreg\s+(\w+)\[', qasm).group(1)

    I2 = np.eye(2, dtype=DTYPE)
    raw = []
    pending = {}

    for line in qasm.strip().splitlines():
        line = line.strip().rstrip(';')
        if not line or any(line.startswith(k) for k in
                           ['OPENQASM', 'include', 'qreg', '//']):
            continue

        m = re.match(rf'^cz\s+{reg}\[(\d+)\],{reg}\[(\d+)\]$', line)
        if m:
            q0, q1 = int(m.group(1)), int(m.group(2))
            for q in [q0, q1]:
                if q in pending:
                    raw.append(('1q', q, pending.pop(q)))
            raw.append(('cz', q0, q1))
            continue

        m = re.match(rf'^u3\(([^)]+)\)\s+{reg}\[(\d+)\]$', line)
        if m:
            params = [parse_angle(p) for p in m.group(1).split(',')]
            q = int(m.group(2))
            pending[q] = u3_mat(*params) @ pending.get(q, I2)
            continue

    for q, U in pending.items():
        raw.append(('1q', q, U))

    n_cz = sum(1 for o in raw if o[0] == 'cz')
    n_1q = sum(1 for o in raw if o[0] == '1q')
    print(f"  Parsed: {n_cz} CZ + {n_1q} fused 1q = {len(raw)} total")
    return n, raw


def parse_rzz_u_hqap(qasm_path: str):
    """
    Parser for P9 (HQAP): RZZ(-π/2) + U gates.

    The 'u' gate in HQAP circuits is identical to U3 — same 3-parameter
    convention as Qiskit's U gate. All RZZ angles are exactly -π/2.

    IMPORTANT: Do not try to shortcut this with Ising SA. The U gates
    are not small perturbations — they fundamentally reshape the output.
    """
    qasm = open(qasm_path).read()
    n = int(re.search(r'qreg\s+q\[(\d+)\]', qasm).group(1))
    I2 = np.eye(2, dtype=DTYPE)
    raw = []
    pending = {}

    rzz_count = 0
    for line in qasm.strip().splitlines():
        line = line.strip().rstrip(';')
        if not line or any(line.startswith(k) for k in
                           ['OPENQASM', 'include', 'qreg', '//']):
            continue

        m = re.match(r'^rzz\(([^)]+)\)\s+q\[(\d+)\],q\[(\d+)\]$', line)
        if m:
            angle = parse_angle(m.group(1))
            q0, q1 = int(m.group(2)), int(m.group(3))
            for q in [q0, q1]:
                if q in pending:
                    raw.append(('1q', q, pending.pop(q)))
            raw.append(('rzz', q0, q1, angle))
            rzz_count += 1
            continue

        # 'u' gate: same as u3
        m = re.match(r'^u\(([^)]+)\)\s+q\[(\d+)\]$', line)
        if m:
            params = [parse_angle(p) for p in m.group(1).split(',')]
            q = int(m.group(2))
            pending[q] = u3_mat(*params) @ pending.get(q, I2)
            continue

    for q, U in pending.items():
        raw.append(('1q', q, U))

    n_1q = sum(1 for o in raw if o[0] == '1q')
    print(f"  Parsed: {rzz_count} RZZ + {n_1q} fused 1q = {len(raw)} total")
    return n, raw


# ─── MPS Build ────────────────────────────────────────────────────────────────

def build_mps(n: int, ops: list, chi_max: int, circuit_type: str = 'cz') -> list:
    """Build MPS by sequentially applying all gates."""
    tensors = mps_init(n)
    t0 = time.time()

    for i, op in enumerate(ops):
        if op[0] == '1q':
            mps_apply_1q(tensors, op[2], op[1])

        elif op[0] == 'cz':
            q0, q1 = op[1], op[2]
            if abs(q0 - q1) == 1:
                mps_apply_nn(tensors, CZ4, min(q0, q1), chi_max)
            else:
                mps_apply_lr_gate(tensors, q0, q1, CZ4, chi_max)

        elif op[0] == 'rzz':
            q0, q1, angle = op[1], op[2], op[3]
            G = rzz_mat(angle)
            if abs(q0 - q1) == 1:
                mps_apply_nn(tensors, G, min(q0, q1), chi_max)
            else:
                mps_apply_lr_gate(tensors, q0, q1, G, chi_max)

        if (i + 1) % 1000 == 0:
            mc = max_bond_dim(tensors)
            print(f"\r    {i+1}/{len(ops)}  max_χ={mc}  t={time.time()-t0:.0f}s  ",
                  end='', flush=True)

    mc = max_bond_dim(tensors)
    print(f"\n  Built: {time.time()-t0:.1f}s  max_χ={mc}")
    return tensors


# ─── Multi-Chi Cross-Validation ───────────────────────────────────────────────

def solve_with_cross_validation(
    qasm_path: str,
    chi_values: list,
    circuit_type: str = 'cz',
    beam_width: int = 500
) -> str:
    """
    Run MPS at multiple χ values. Collect all candidate bitstrings,
    then evaluate each under the best (highest-χ) MPS to find the winner.

    The key insight: a candidate that scores well under a better MPS is
    more reliable. The cross-χ score difference tells you how much you
    can trust each χ level.
    """
    print(f"\n  Circuit: {os.path.basename(qasm_path)}")
    print(f"  χ values: {chi_values}  |  beam width: {beam_width}")

    # Parse once
    if circuit_type == 'rzz':
        n, ops = parse_rzz_u_hqap(qasm_path)
    else:
        n, ops = parse_cz_u3_alltoall(qasm_path)

    all_candidates = {}   # bitstring → best logP seen so far
    best_tensors = None
    best_chi = 0

    for chi in chi_values:
        print(f"\n{'='*58}")
        print(f"  χ = {chi}")
        print(f"{'='*58}")

        tensors = build_mps(n, ops, chi, circuit_type)

        # Decode
        pk_lr, pk_rl, best, common = bidirectional_decode(
            tensors, W=beam_width, top_k=50)

        agree = pk_lr == pk_rl
        bits_match = sum(a == b for a, b in zip(pk_lr, pk_rl))
        print(f"  LR: {pk_lr}  logP={pk_lr and mps_logprob(tensors,pk_lr):.3f}")
        print(f"  RL: {pk_rl}")
        print(f"  LR==RL: {'✓ IDENTICAL' if agree else f'✗ ({bits_match}/{n})'}")
        print(f"  Common top-50: {len(common)}")

        # Perturbation test
        final, final_lp = perturbation_test(tensors, best)
        changed = 'improved' if final != best else 'local max ✓'
        print(f"  After perturbation: {final}  logP={final_lp:.4f}  [{changed}]")

        # Collect candidates
        for bs in [pk_lr, pk_rl, final] + list(common):
            lp = mps_logprob(tensors, bs)
            if bs not in all_candidates or lp > all_candidates[bs]:
                all_candidates[bs] = lp

        # Keep best tensors for final scoring
        if chi >= best_chi:
            best_chi = chi
            best_tensors = tensors

    # Score all candidates under best-χ MPS
    print(f"\n{'='*58}")
    print(f"  Final scoring under χ={best_chi} MPS")
    print(f"{'='*58}")

    scored = {bs: mps_logprob(best_tensors, bs) for bs in all_candidates}
    ranked = sorted(scored.items(), key=lambda x: -x[1])

    print(f"\n  Top 5 candidates (χ={best_chi} scores):")
    for rank, (bs, lp) in enumerate(ranked[:5]):
        print(f"  {rank+1}. {bs}  logP={lp:.4f}")

    # Final perturbation on winner
    winner_bs = ranked[0][0]
    winner, winner_lp = perturbation_test(best_tensors, winner_bs)
    changed = 'improved' if winner != winner_bs else 'local max ✓'
    print(f"\n  Final answer: {winner}  logP={winner_lp:.4f}  [{changed}]")

    print(f"\n{'='*58}")
    print(f"  PEAK:    {winner}")
    print(f"  REVERSE: {winner[::-1]}")
    print(f"{'='*58}")

    return winner


# ─── Auto-Config ──────────────────────────────────────────────────────────────

def detect_config(qasm_path: str):
    """Detect problem type and return chi_values and circuit_type."""
    stem = os.path.basename(qasm_path).upper()

    if 'P9' in stem or 'HQAP' in stem:
        print("  Detected: P9 HQAP — using RZZ parser")
        print("  NOTE: Ising SA gives wrong answer here. Using MPS.")
        return PROBLEM_CONFIGS['P9']['chi_values'], 'rzz'
    elif 'P5' in stem or 'GRANITE' in stem:
        print("  Detected: P5 Granite Summit — all-to-all 44q")
        return PROBLEM_CONFIGS['P5']['chi_values'], 'cz'
    elif 'P6' in stem or 'TITAN' in stem:
        print("  Detected: P6 Titan Pinnacle — all-to-all 62q")
        return PROBLEM_CONFIGS['P6']['chi_values'], 'cz'
    else:
        print("  Unknown circuit — using default [16, 24], CZ")
        return [16, 24], 'cz'


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: Approximate MPS for all-to-all circuits')
    parser.add_argument('--qasm', required=True)
    parser.add_argument('--chi', nargs='+', type=int, default=[],
                        help='Bond dimensions, e.g. --chi 16 24 32')
    parser.add_argument('--beam', type=int, default=500,
                        help='Beam search width (default: 500)')
    args = parser.parse_args()

    auto_chi, auto_type = detect_config(args.qasm)
    chi_values = args.chi if args.chi else auto_chi

    print('=' * 60)
    print(f'  Stage 3 — Approximate MPS  [χ={chi_values}]')
    print('=' * 60)

    solve_with_cross_validation(
        args.qasm,
        chi_values=chi_values,
        circuit_type=auto_type,
        beam_width=args.beam
    )


if __name__ == '__main__':
    main()
