# Quantum Peak Finder — Complete Challenge Solutions

> **10 quantum circuits. 10 peak bitstrings. One classical simulator built from scratch.**

This repo contains my complete solution to a quantum reverse-engineering challenge: given 10 circuits in QASM 2.0 format, find the "peak bitstring" — the computational basis state whose measurement probability dominates all others by orders of magnitude.

I wrote a pure-NumPy Matrix Product State (MPS) simulator from scratch (no Qiskit, no quantum libraries), which ended up being the right tool for nearly every problem. The approach scales from a 4-qubit toy circuit all the way to a 62-qubit all-to-all entangling monster.

---

## Results at a Glance

| # | Name | Qubits | Connectivity | 2Q Gates | Peak Bitstring | Confidence |
|---|------|--------|-------------|----------|---------------|------------|
| 1 | Little Peak | 4 | None | 0 | `1001` | ⭐⭐⭐⭐⭐ exact |
| 2 | Swift Rise | 28 | Ring | 210 | `0011100001101100011011010011` | ⭐⭐⭐⭐⭐ |
| 3 | Sharp Peak | 44 | Ring | 186 | `10001101010101010000011111001101000100011010` | ⭐⭐⭐⭐⭐ |
| 4 | Golden Mountain | 48 | Ring | 5096 | `100001010111001101010011110011100100010010100101` | ⭐⭐⭐⭐⭐ |
| 5 | Granite Summit | 44 | All-to-All | 1892 | `10111011110110100010011010001111110010110110` | ⭐⭐⭐⭐ |
| 6 | Titan Pinnacle | 62 | All-to-All | 3494 | `10110010100001110111001110110010111111001101001110101010100110` | ⭐⭐⭐⭐⭐ |
| 7 | Heavy Hex 1275 | 45 | IBM Heavy Hex | 1275 | `011000011111011011000110111111111000010001000` | ⭐⭐⭐⭐⭐ |
| 8 | Grid 888 iSwap | 40 | Google Willow Grid | 888 | `0101111100100000110001100101101101001110` | ⭐⭐⭐⭐ |
| 9 | HQAP 1917 | 56 | All-to-All | 1917 | `01100111001010001001110011011001010100010010110100001001` | ⭐⭐⭐ |
| 10 | Heavy Hex 4020 | 49 | IBM Heavy Hex | 4020 | `1000100111111110110100001110101101101000111010011` | ⭐⭐⭐⭐ |

> **Note on bit ordering:** The challenge accepts both a bitstring and its bitwise reverse as correct answers. The two output conventions (statevector MSB-first vs. MPS left-to-right) differ but both are valid submissions.

---

## Repository Structure

```
quantum-peak-finder/
├── README.md                        ← you are here
├── solutions/
│   ├── core_mps.py                  ← shared MPS engine (pure NumPy)
│   ├── stage1_exact.py              ← P1, P2: exact statevector simulation
│   ├── stage2_mps.py                ← P3, P4, P7, P8, P10: MPS + beam search
│   ├── stage3_heuristic.py          ← P5, P6, P9: approximate MPS + cross-χ validation
│   └── run_all.py                   ← master runner, saves results.json
├── circuits/                        ← place your .qasm files here
└── docs/
    ├── ANALYSIS.md                  ← per-problem technical writeup
    └── TELECOM_QUANTUM.md           ← background on quantum tech & telecom connections
```

---

## The Core Problem

Every circuit produces a final quantum state:

$$|\psi\rangle = U |0\rangle^{\otimes n} = \sum_{x \in \{0,1\}^n} \alpha_x |x\rangle$$

We need to find $x^* = \arg\max_x |\alpha_x|^2$, the bitstring with the largest measurement probability.

The difficulty: exact simulation requires storing all $2^n$ amplitudes. For $n = 56$ (P9), that's $7 \times 10^{16}$ complex numbers — roughly **576 petabytes** of RAM. Something smarter is needed.

---

## Three-Stage Strategy

### Stage 1 — Exact Simulation (P1, P2)

Small enough to simulate directly. P1 is a product state (no entangling gates), solved analytically. P2 uses a memory-optimized statevector simulation with reshape-based gate application.

```bash
python solutions/stage1_exact.py --qasm circuits/P1_little_peak.qasm
python solutions/stage1_exact.py --qasm circuits/P2_swift_rise.qasm
```

### Stage 2 — MPS Simulation (P3, P4, P7, P8, P10)

These circuits have structured, sparse connectivity (ring, grid, IBM Heavy Hex, Google Willow). By the **area law of entanglement entropy**, the quantum state can be approximated efficiently with Matrix Product States at bond dimension χ ≪ 2^(n/2).

```bash
python solutions/stage2_mps.py --qasm circuits/P3_sharp_peak.qasm --chi 256
python solutions/stage2_mps.py --qasm circuits/P7_heavy_hex_1275.qasm --chi 64
python solutions/stage2_mps.py --qasm circuits/P8_grid_888_iswap.qasm --chi 80
```

### Stage 3 — Approximate MPS (P5, P6, P9)

All-to-all connectivity means the area law doesn't apply cleanly. I use MPS with lower χ, compensating with multi-χ cross-validation: run at several bond dimensions, then evaluate all candidates under the best available MPS to pick the most reliable answer.

```bash
python solutions/stage3_heuristic.py --qasm circuits/P5_granite_summit.qasm
python solutions/stage3_heuristic.py --qasm circuits/P6_titan_pinnacle.qasm
python solutions/stage3_heuristic.py --qasm circuits/P9_hqap_1917.qasm
```

---

## Installation

```bash
# All you need
pip install numpy scipy

# Optional — only needed if you want to verify results with Qiskit
pip install qiskit qiskit-aer
```

No quantum computing library is required for the simulator itself. Everything runs on pure NumPy.

---

## Run Everything

```bash
# Put all .qasm files in ./circuits/
python solutions/run_all.py --dir ./circuits

# Dry run first to verify file locations
python solutions/run_all.py --dir ./circuits --dry_run

# Run specific problems only
python solutions/run_all.py --dir ./circuits --problems P7 P8 P10
```

Results are saved to `results.json` with bitstrings, log-probabilities, timing, and confidence metadata.

---

## Key Technical Decisions

**Why a custom MPS instead of Qiskit Aer?**
The compute environment didn't have Qiskit available. The custom NumPy implementation turned out to be cleaner anyway — full control over χ, truncation thresholds, and gate ordering. It handles all gate types across all 10 problems from a single codebase.

**Why bidirectional beam search instead of sampling?**
For strongly peaked states, beam search is far more efficient than sampling. Running from both ends (left-to-right and right-to-left) and checking that both directions agree gives a built-in consistency check. When LR == RL with beam width W=1000, that's about as confident as you can be without exact simulation.

**The P9 warning:**
For HQAP 1917, I initially tried Ising Simulated Annealing (since all 1917 RZZ gates have the same angle −π/2, making it look like a classical Ising model). SA confidently returned `111...1` (all ones). That answer is wrong — it completely ignores the 3890 single-qubit U gates that determine the actual quantum interference pattern. Always simulate the full circuit.

**Bit ordering:**
- `stage1_exact.py` outputs bitstrings in Qiskit convention: `q[n-1]q[n-2]...q[1]q[0]`
- `stage2_mps.py` and `stage3_heuristic.py` output in MPS left-to-right order: `q[0]q[1]...q[n-1]`

Since the challenge accepts both a bitstring and its reverse, this distinction doesn't affect submission correctness.

---

## Validation Protocol

Each answer was accepted only after passing a stack of checks:

| Check | What it means |
|-------|--------------|
| **LR = RL** | Left-to-right and right-to-left beam search return the same bitstring |
| **Common top-50** | Most top-50 candidates from both directions overlap |
| **Local maximum** | No single-bit flip improves the log-probability (perturbation test) |
| **Cross-χ** | Candidate scores highest under the best available MPS |
| **Multi-run** | Results are stable across different beam widths and seeds |

P7 (Heavy Hex 1275) passes all five checks and is the most reliable result.

---

## Further Reading

- [docs/ANALYSIS.md](docs/ANALYSIS.md) — Detailed per-problem walkthrough
- [docs/TELECOM_QUANTUM.md](docs/TELECOM_QUANTUM.md) — Quantum technology and telecommunications background, including connections to CV-QKD, quantum networks, and IBM/Google hardware

---

## License

MIT — use freely, attribution appreciated.
