"""
run_all.py — Master runner for all 10 quantum peak problems
===========================================================

Runs all 10 problems sequentially, saves results to JSON, and prints
a summary table with peak bitstrings ready for submission.

Usage:
    python run_all.py --dir /path/to/qasm/files
    python run_all.py --dir . --dry_run        # check setup without running
    python run_all.py --dir . --problems P1 P7  # specific problems only
    python run_all.py --dir . --output my_results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Problem Configuration ────────────────────────────────────────────────────

PROBLEMS = {
    'P1': {
        'file': 'P1_little_peak.qasm',
        'stage': 1,
        'chi': None,
        'gate': None,
        'note': '4 qubits, no entangling gates — analytic product state',
    },
    'P2': {
        'file': 'P2_swift_rise.qasm',
        'stage': 1,
        'chi': None,
        'gate': None,
        'note': '28 qubits, ring — exact statevector, needs ~2 GB RAM',
    },
    'P3': {
        'file': 'P3_sharp_peak.qasm',
        'stage': 2,
        'chi': 256,
        'gate': 'cz',
        'note': 'Ring 44q, 186 CZ — χ=256 is near-exact',
    },
    'P4': {
        'file': 'P4_golden_mountain.qasm',
        'stage': 2,
        'chi': 32,
        'gate': 'cz',
        'note': 'Ring 48q, 5096 CZ — deep circuit, beam search is the key',
    },
    'P5': {
        'file': 'P5_granite_summit.qasm',
        'stage': 3,
        'chi': [24, 32],
        'gate': 'cz',
        'note': 'All-to-all 44q, 1892 CZ — approximate, χ=32 LR=RL confirmed',
    },
    'P6': {
        'file': 'P6_titan_pinnacle.qasm',
        'stage': 3,
        'chi': [16, 24],
        'gate': 'cz',
        'note': 'All-to-all 62q, 3494 CZ — χ=24 beats χ=16 by +91.6 nats',
    },
    'P7': {
        'file': 'P7_heavy_hex_1275.qasm',
        'stage': 2,
        'chi': 64,
        'gate': 'cz',
        'note': 'IBM Heavy Hex 45q, 1275 CZ — most reliable result',
    },
    'P8': {
        'file': 'P8_grid_888_iswap.qasm',
        'stage': 2,
        'chi': 80,
        'gate': 'iswap',
        'note': 'Google Willow Grid 40q, 888 iSWAP — χ=80 needed (χ=64 wrong)',
    },
    'P9': {
        'file': 'P9_hqap_1917.qasm',
        'stage': 3,
        'chi': [16, 24],
        'gate': 'rzz',
        'note': 'HQAP 56q, 1917 RZZ — hardest; Ising SA gives wrong answer!',
    },
    'P10': {
        'file': 'P10_heavy_hex_4020.qasm',
        'stage': 2,
        'chi': 48,
        'gate': 'cz',
        'note': 'IBM Heavy Hex 49q, 4020 CZ — χ=48 needed (χ=32 wrong)',
    },
}

# Known answers from verified runs
KNOWN_ANSWERS = {
    'P1':  '1001',
    'P2':  '0011100001101100011011010011',
    'P3':  '10001101010101010000011111001101000100011010',
    'P4':  '100001010111001101010011110011100100010010100101',
    'P5':  '10111011110110100010011010001111110010110110',
    'P6':  '10110010100001110111001110110010111111001101001110101010100110',
    'P7':  '011000011111011011000110111111111000010001000',
    'P8':  '0101111100100000110001100101101101001110',
    'P9':  '01100111001010001001110011011001010100010010110100001001',
    'P10': '1000100111111110110100001110101101101000111010011',
}

CONFIDENCE = {
    'P1': '5/5 — analytic exact',
    'P2': '5/5 — 3 chi values agree',
    'P3': '5/5 — 3 chi values agree, logP diff=0',
    'P4': '5/5 — W=50/200/500 all identical',
    'P5': '4/5 — LR=RL, local max confirmed',
    'P6': '5/5 — +91.6 nats over chi=16, LR=RL',
    'P7': '5/5 — best result in challenge',
    'P8': '4/5 — +8.1 nats over chi=64, LR=RL',
    'P9': '3/5 — LR≈RL (45/56 bits), hardest problem',
    'P10': '4/5 — +20.7 nats over chi=32, LR=RL',
}


# ─── File Finding ─────────────────────────────────────────────────────────────

def find_qasm(directory: str, filename: str):
    """Case-insensitive file search."""
    for p in Path(directory).glob('*.qasm'):
        if p.name.lower() == filename.lower():
            return str(p)
    return None


# ─── Problem Runner ───────────────────────────────────────────────────────────

def run_problem(pid: str, config: dict, qasm_dir: str) -> dict:
    """Run a single problem and return a result dict."""
    qasm_path = find_qasm(qasm_dir, config['file'])
    if not qasm_path:
        return {
            'problem': pid,
            'status': 'FILE_NOT_FOUND',
            'peak': None, 'reverse': None,
            'known': KNOWN_ANSWERS.get(pid),
            'match': None,
            'elapsed_s': 0,
            'error': f"{config['file']} not found in {qasm_dir}",
        }

    print(f"\n{'='*65}")
    print(f"  {pid}  —  {config['note']}")
    print(f"{'='*65}")

    t0 = time.time()
    peak = None
    status = 'FAILED'
    error = None

    try:
        stage = config['stage']
        sys.path.insert(0, str(Path(__file__).parent))

        if stage == 1:
            from stage1_exact import parse_and_simulate
            peak = parse_and_simulate(qasm_path)

        elif stage == 2:
            from stage2_mps import run_mps
            peak = run_mps(
                qasm_path,
                chi_max=config['chi'],
                gate_type=config['gate'],
                beam_width=500
            )

        elif stage == 3:
            from stage3_heuristic import solve_with_cross_validation
            ctype = config['gate'] if config['gate'] != 'cz' else 'cz'
            if config['gate'] == 'rzz':
                ctype = 'rzz'
            peak = solve_with_cross_validation(
                qasm_path,
                chi_values=config['chi'],
                circuit_type=ctype,
                beam_width=500
            )

        status = 'SUCCESS' if peak else 'NO_RESULT'

    except MemoryError:
        error = 'OUT_OF_MEMORY — try reducing chi or use a machine with more RAM'
        status = 'OOM'
        print(f"\n  {error}")

    except Exception as e:
        error = str(e)
        status = 'ERROR'
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0

    # Check against known answer (accept forward or reverse)
    known = KNOWN_ANSWERS.get(pid)
    match = None
    if peak and known:
        if peak == known or peak == known[::-1]:
            match = 'CORRECT ✓'
        else:
            n = min(len(peak), len(known))
            bits_fwd = sum(a == b for a, b in zip(peak, known))
            bits_rev = sum(a == b for a, b in zip(peak, known[::-1]))
            best_bits = max(bits_fwd, bits_rev)
            match = f'DIFFERS ({best_bits}/{n} bits match best direction)'

    return {
        'problem': pid,
        'status': status,
        'peak': peak,
        'reverse': peak[::-1] if peak else None,
        'known': known,
        'match': match,
        'confidence': CONFIDENCE.get(pid, '?'),
        'stage': stage,
        'elapsed_s': round(elapsed, 2),
        'error': error,
        'timestamp': datetime.now().isoformat(),
    }


# ─── Summary Printer ──────────────────────────────────────────────────────────

def print_summary(results: list):
    print(f"\n{'='*75}")
    print(f"  SUMMARY  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*75}")
    print(f"  {'ID':<6} {'Status':<15} {'Time':>7}  {'Confidence':<25} Peak (first 24 bits)")
    print(f"  {'-'*70}")

    n_success = 0
    for r in results:
        icon = '✓' if r['status'] == 'SUCCESS' else '✗'
        preview = (r['peak'][:24] + '...') if r['peak'] else 'N/A'
        conf = r.get('confidence', '?')[:20]
        match_str = f"  [{r['match']}]" if r.get('match') else ''
        print(f"  {r['problem']:<6} {icon} {r['status']:<13} {r['elapsed_s']:>7.1f}s  {conf:<25} {preview}{match_str}")
        if r['status'] == 'SUCCESS':
            n_success += 1

    print(f"  {'-'*70}")
    print(f"  Solved: {n_success}/{len(results)} problems")
    print(f"{'='*75}\n")

    print("Peak bitstrings for submission (forward and reverse both accepted):")
    print("-" * 70)
    for r in results:
        if r['peak']:
            print(f"  {r['problem']}: {r['peak']}")
            print(f"  {r['problem']} (rev): {r['reverse']}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run all 10 quantum peak problems')
    parser.add_argument('--dir', default='.', help='Directory with .qasm files')
    parser.add_argument('--problems', nargs='+', default=None,
                        help='Problems to run, e.g. P1 P7 P10 (default: all)')
    parser.add_argument('--output', default='results.json',
                        help='Output JSON file (default: results.json)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print config and check files without running')
    args = parser.parse_args()

    targets = args.problems or list(PROBLEMS.keys())
    unknown = [p for p in targets if p not in PROBLEMS]
    if unknown:
        print(f"Unknown problem IDs: {unknown}")
        print(f"Valid IDs: {list(PROBLEMS.keys())}")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run — checking configuration:\n")
        for pid in targets:
            cfg = PROBLEMS[pid]
            found = find_qasm(args.dir, cfg['file'])
            status = '✓ found' if found else '✗ NOT FOUND'
            chi_str = str(cfg['chi']) if cfg['chi'] else 'exact'
            print(f"  {pid}: stage={cfg['stage']}  chi={chi_str:<12} {status}")
            print(f"       {cfg['note']}")
        return

    results = []
    for pid in targets:
        result = run_problem(pid, PROBLEMS[pid], args.dir)
        results.append(result)

    # Save JSON
    output_path = Path(args.dir) / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    print_summary(results)


if __name__ == '__main__':
    main()
