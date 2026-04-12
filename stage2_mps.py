"""
stage2_mps.py — MPS simulation for structured-connectivity circuits
===================================================================

Handles:
  P3  — Ring 44q,  186 CZ,  χ=256
  P4  — Ring 48q, 5096 CZ,  χ=32  (deep circuit — beam search is key)
  P7  — IBM Heavy Hex 45q, 1275 CZ, χ=64
  P8  — Google Willow Grid 40q, 888 iSWAP, χ=80
  P10 — IBM Heavy Hex 49q, 4020 CZ, χ=48

These circuits have structured, sparse connectivity. The area law of
entanglement entropy applies, meaning MPS at moderate χ is highly accurate.

Bit ordering note:
  Output bits are in MPS site order (q[0], q[1], ..., q[n-1]).
  The challenge accepts both this and its reverse as correct.

Usage:
    python stage2_mps.py --qasm P3_sharp_peak.qasm --chi 256
    python stage2_mps.py --qasm P7_heavy_hex_1275.qasm --chi 64
    python stage2_mps.py --qasm P8_grid_888_iswap.qasm --chi 80
    python stage2_mps.py --qasm P10_heavy_hex_4020.qasm --chi 48
"""

import argparse
import os
import re
import sys
import time
import numpy as np
from core_mps import (
    mps_init, mps_apply_1q, mps_apply_nn, mps_apply_lr_gate,
    fuse_1q_gates, bidirectional_decode, perturbation_test, mps_logprob,
    max_bond_dim, bond_profile,
    CZ4, ISWAP4, u3_mat, parse_angle, DTYPE
)

# Empirically validated χ values per problem
RECOMMENDED_CHI = {
    'P3': 256,   # ring 44q — χ=64 already converges, 256 is essentially exact
    'P4': 32,    # ring 48q, deep — beam search matters more than χ here
    'P7': 64,    # heavy-hex 45q — sparse graph, χ=64 is near-exact
    'P8': 80,    # grid 40q iSwap — χ=64 not converged, χ=80 gives +8 nats
    'P10': 48,   # heavy-hex 49q — χ=32 not converged, χ=48 gives +21 nats
}


# ─── QASM Parsers ─────────────────────────────────────────────────────────────

def parse_cz_u3(qasm_path: str):
    """
    Parser for circuits with CZ + U3 gates (P3, P4, P7, P10).
    Handles arbitrary register names.
    Returns: (n_qubits, fused_ops_list)
    """
    qasm = open(qasm_path).read()
    n = int(re.search(r'qreg\s+\w+\[(\d+)\]', qasm).group(1))
    I2 = np.eye(2, dtype=DTYPE)
    raw = []
    pending = {}  # qubit → accumulated 1q matrix (for fusion)

    for line in qasm.strip().splitlines():
        line = line.strip().rstrip(';')
        if not line or any(line.startswith(k) for k in
                           ['OPENQASM', 'include', 'qreg', '//']):
            continue

        m = re.match(r'^cz\s+\w+\[(\d+)\],\w+\[(\d+)\]$', line)
        if m:
            q0, q1 = int(m.group(1)), int(m.group(2))
            for q in [q0, q1]:
                if q in pending:
                    raw.append(('1q', q, pending.pop(q)))
            raw.append(('cz', q0, q1))
            continue

        m = re.match(r'^u3\(([^)]+)\)\s+\w+\[(\d+)\]$', line)
        if m:
            params = [parse_angle(p) for p in m.group(1).split(',')]
            q = int(m.group(2))
            pending[q] = u3_mat(*params) @ pending.get(q, I2)
            continue

    for q, U in pending.items():
        raw.append(('1q', q, U))

    n_cz = sum(1 for o in raw if o[0] == 'cz')
    n_1q = sum(1 for o in raw if o[0] == '1q')
    print(f"  Parsed: {n_cz} CZ + {n_1q} fused 1q = {len(raw)} total ops")
    return n, raw


def parse_iswap_u3(qasm_path: str):
    """
    Parser for P8 (Google Willow Grid): iSWAP + U3 gates.

    The QASM file defines iSWAP as a macro on one line:
      gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }

    We strip this definition before parsing instructions, then handle
    'iswap q[a],q[b]' as a primitive with the known 4×4 matrix.

    Verified: the decomposition gives exactly
    iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    """
    text = open(qasm_path).read()
    # Strip the gate definition (it's on one line with {...})
    text = re.sub(r'^gate\s+iswap[^}]+\}\s*$', '', text, flags=re.MULTILINE)

    n = int(re.search(r'qreg\s+\w+\[(\d+)\]', text).group(1))
    I2 = np.eye(2, dtype=DTYPE)
    raw = []
    pending = {}

    for line in text.strip().splitlines():
        line = line.strip().rstrip(';')
        if not line or any(line.startswith(k) for k in
                           ['OPENQASM', 'include', 'qreg', '//']):
            continue

        m = re.match(r'^iswap\s+q\[(\d+)\],q\[(\d+)\]$', line)
        if m:
            q0, q1 = int(m.group(1)), int(m.group(2))
            for q in [q0, q1]:
                if q in pending:
                    raw.append(('1q', q, pending.pop(q)))
            raw.append(('iswap', q0, q1))
            continue

        m = re.match(r'^u3\(([^)]+)\)\s+q\[(\d+)\]$', line)
        if m:
            params = [parse_angle(p) for p in m.group(1).split(',')]
            q = int(m.group(2))
            pending[q] = u3_mat(*params) @ pending.get(q, I2)
            continue

    for q, U in pending.items():
        raw.append(('1q', q, U))

    n_is = sum(1 for o in raw if o[0] == 'iswap')
    n_1q = sum(1 for o in raw if o[0] == '1q')
    print(f"  Parsed: {n_is} iSWAP + {n_1q} fused 1q = {len(raw)} total ops")
    return n, raw


# ─── Simulation Pipeline ──────────────────────────────────────────────────────

def run_mps(
    qasm_path: str,
    chi_max: int,
    gate_type: str = 'cz',
    beam_width: int = 500,
    verbose: bool = True
) -> str:
    """
    Full MPS pipeline:
      1. Parse QASM and fuse single-qubit gates
      2. Build MPS by applying all gates
      3. Decode with bidirectional beam search
      4. Validate with perturbation test

    Returns the peak bitstring (in MPS site order q[0]...q[n-1]).
    """
    # Step 1: Parse
    if gate_type == 'iswap':
        n, ops = parse_iswap_u3(qasm_path)
        g2_gate = ISWAP4
        g2_key = 'iswap'
    else:
        n, ops = parse_cz_u3(qasm_path)
        g2_gate = CZ4
        g2_key = 'cz'

    # Step 2: Build MPS
    print(f"\n  Building MPS: n={n} qubits, χ_max={chi_max}, {len(ops)} ops")
    tensors = mps_init(n)
    t0 = time.time()

    for i, op in enumerate(ops):
        if op[0] == '1q':
            mps_apply_1q(tensors, op[2], op[1])
        elif op[0] in ('cz', 'iswap'):
            q0, q1 = op[1], op[2]
            if abs(q0 - q1) == 1:
                mps_apply_nn(tensors, g2_gate, min(q0, q1), chi_max)
            else:
                mps_apply_lr_gate(tensors, q0, q1, g2_gate, chi_max)

        if verbose and (i + 1) % 1000 == 0:
            mc = max_bond_dim(tensors)
            print(f"\r    {i+1}/{len(ops)}  max_χ={mc}  t={time.time()-t0:.0f}s  ",
                  end='', flush=True)

    mc = max_bond_dim(tensors)
    print(f"\n  Done: {time.time()-t0:.1f}s  max_χ={mc}")
    print(f"  Bond profile: {bond_profile(tensors)}")

    # Step 3: Bidirectional beam search
    print(f"\n  Bidirectional beam search (W={beam_width})...")
    pk_lr, pk_rl, best, common = bidirectional_decode(
        tensors, W=beam_width, top_k=50)

    agree = pk_lr == pk_rl
    bits_match = sum(a == b for a, b in zip(pk_lr, pk_rl))
    print(f"  LR: {pk_lr}")
    print(f"  RL: {pk_rl}")
    print(f"  LR == RL: {'✓ IDENTICAL' if agree else f'✗ ({bits_match}/{n} bits)'}")
    print(f"  Common top-50: {len(common)}")

    # Step 4: Perturbation test
    print(f"\n  Perturbation test...")
    final, final_lp = perturbation_test(tensors, best)
    changed = 'improved' if final != best else 'local maximum ✓'
    print(f"  Result: {final}  logP={final_lp:.4f}  [{changed}]")

    print(f"\n{'='*60}")
    print(f"  PEAK:    {final}")
    print(f"  REVERSE: {final[::-1]}")
    print(f"{'='*60}")

    return final


# ─── Auto-Config ──────────────────────────────────────────────────────────────

def detect_config(qasm_path: str):
    """Guess chi and gate type from filename."""
    stem = os.path.basename(qasm_path).upper()
    chi = 64
    gate = 'cz'

    if 'P3' in stem or 'SHARP' in stem:
        chi = RECOMMENDED_CHI['P3']
    elif 'P4' in stem or 'GOLDEN' in stem:
        chi = RECOMMENDED_CHI['P4']
    elif 'P7' in stem or ('HEX' in stem and '1275' in stem):
        chi = RECOMMENDED_CHI['P7']
    elif 'P8' in stem or 'ISWAP' in stem or 'GRID' in stem:
        chi = RECOMMENDED_CHI['P8']
        gate = 'iswap'
    elif 'P10' in stem or ('HEX' in stem and '4020' in stem):
        chi = RECOMMENDED_CHI['P10']

    return chi, gate


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: MPS simulation for structured circuits')
    parser.add_argument('--qasm', required=True)
    parser.add_argument('--chi', type=int, default=0,
                        help='Bond dimension χ (0 = auto-detect from filename)')
    parser.add_argument('--gate', default='auto',
                        choices=['auto', 'cz', 'iswap'],
                        help='Gate type (auto = detect from filename)')
    parser.add_argument('--beam', type=int, default=500,
                        help='Beam search width (default: 500)')
    args = parser.parse_args()

    auto_chi, auto_gate = detect_config(args.qasm)
    chi = args.chi if args.chi > 0 else auto_chi
    gate = args.gate if args.gate != 'auto' else auto_gate

    print('=' * 60)
    print(f'  Stage 2 — MPS Simulation  [χ={chi}, gate={gate}]')
    print('=' * 60)

    run_mps(args.qasm, chi_max=chi, gate_type=gate, beam_width=args.beam)


if __name__ == '__main__':
    main()
