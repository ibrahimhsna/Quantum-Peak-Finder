"""
core_mps.py — Matrix Product State simulator
=============================================

Shared engine used by all three solution stages.
Pure NumPy — no quantum computing libraries required.

Quick reference:
  mps_init(n)                    Initialize |00...0⟩
  mps_apply_1q(tensors, U, k)    Apply 2×2 unitary to qubit k
  mps_apply_nn(tensors, G, k, χ) Apply 4×4 gate to qubits (k, k+1)
  mps_apply_lr_gate(...)         Long-range gate via SWAP chain
  beam_search_lr(tensors, W)     Decode: left-to-right beam search
  beam_search_rl(tensors, W)     Decode: right-to-left beam search
  bidirectional_decode(...)      Run both beams, find intersection
  mps_logprob(tensors, bits)     Compute log P(bitstring)
  perturbation_test(tensors, x)  Check if x is a local maximum

Bit ordering note:
  Beam search outputs bits in site order: q[0], q[1], ..., q[n-1].
  Qiskit convention is reversed: q[n-1], ..., q[1], q[0].
  Both are valid for submission since the challenge accepts forward
  and reversed bitstrings equally.
"""

import numpy as np
from numpy.linalg import svd
from typing import List, Tuple, Set, Optional

DTYPE = np.complex64


# ─── Standard Gate Matrices ──────────────────────────────────────────────────

# Two-qubit gates (4×4)
CZ4    = np.diag([1, 1, 1, -1]).astype(DTYPE)
SWAP4  = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=DTYPE)
ISWAP4 = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=DTYPE)

# Single-qubit gates (2×2)
X_MAT  = np.array([[0, 1], [1, 0]], dtype=DTYPE)
H_MAT  = np.array([[1, 1], [1, -1]], dtype=DTYPE) / np.sqrt(2)
S_MAT  = np.array([[1, 0], [0, 1j]], dtype=DTYPE)
SX_MAT = np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=DTYPE) * 0.5


# ─── Parametric Gate Constructors ────────────────────────────────────────────

def u3_mat(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    General single-qubit SU(2) rotation — the most common gate in these circuits.
    Also used for the 'u' gate in HQAP circuits (same convention as Qiskit U3).

    U3(θ,φ,λ) = [[cos(θ/2),         -e^{iλ}sin(θ/2)],
                  [e^{iφ}sin(θ/2),   e^{i(φ+λ)}cos(θ/2)]]
    """
    return np.array([
        [np.cos(theta/2),                -np.exp(1j*lam) * np.sin(theta/2)],
        [np.exp(1j*phi) * np.sin(theta/2), np.exp(1j*(phi+lam)) * np.cos(theta/2)],
    ], dtype=DTYPE)


def rz_mat(angle: float) -> np.ndarray:
    """RZ(θ) = diag(e^{-iθ/2}, e^{+iθ/2})"""
    return np.array([
        [np.exp(-1j*angle/2), 0],
        [0, np.exp(+1j*angle/2)],
    ], dtype=DTYPE)


def ry_mat(angle: float) -> np.ndarray:
    """RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]"""
    c, s = np.cos(angle/2), np.sin(angle/2)
    return np.array([[c, -s], [s, c]], dtype=DTYPE)


def rzz_mat(angle: float) -> np.ndarray:
    """
    RZZ(θ) = exp(-iθ/2 · Z⊗Z)
           = diag(e^{-iθ/2}, e^{+iθ/2}, e^{+iθ/2}, e^{-iθ/2})

    For P9: angle = -π/2, giving:
    RZZ(-π/2) = diag(e^{+iπ/4}, e^{-iπ/4}, e^{-iπ/4}, e^{+iπ/4})
    """
    return np.diag([
        np.exp(-1j * angle / 2),
        np.exp(+1j * angle / 2),
        np.exp(+1j * angle / 2),
        np.exp(-1j * angle / 2),
    ]).astype(DTYPE)


def parse_angle(s: str) -> float:
    """Parse angle string, handling 'pi' and expressions like '-pi/2'."""
    return float(eval(s.strip().replace('pi', str(np.pi))))


# ─── MPS Initialization ──────────────────────────────────────────────────────

def mps_init(n: int) -> List[np.ndarray]:
    """
    Initialize the n-qubit state |00...0⟩ as an MPS.
    Each tensor has shape (1, 2, 1) — product state, no entanglement.
    """
    tensors = []
    for _ in range(n):
        T = np.zeros((1, 2, 1), dtype=DTYPE)
        T[0, 0, 0] = 1.0  # amplitude for |0⟩ = 1
        tensors.append(T)
    return tensors


# ─── Gate Application ─────────────────────────────────────────────────────────

def mps_apply_1q(tensors: List[np.ndarray], U: np.ndarray, site: int) -> None:
    """
    Apply a 2×2 unitary U to site k. In-place, no truncation.
    Cost: O(χ²) — one matrix multiplication per bond.
    """
    T = tensors[site]  # shape (χ_L, 2, χ_R)
    new = np.empty_like(T)
    new[:, 0, :] = U[0, 0] * T[:, 0, :] + U[0, 1] * T[:, 1, :]
    new[:, 1, :] = U[1, 0] * T[:, 0, :] + U[1, 1] * T[:, 1, :]
    tensors[site] = new


def mps_apply_nn(
    tensors: List[np.ndarray],
    G4: np.ndarray,
    site: int,
    chi_max: int
) -> None:
    """
    Apply a 4×4 unitary G4 to neighboring sites (site, site+1).

    Gate convention: G4[i'*2+j', i*2+j] where i,j ∈ {0,1} are
    input qubit values and i',j' are output values.

    Steps:
      1. Contract tensors into Θ matrix: (χ_L · 2) × (2 · χ_R)
      2. Apply gate via einsum on (i,j) physical indices
      3. SVD and truncate to chi_max
      4. Split back: absorb √σ into both tensors

    The truncation is the only approximation step. Error = Σ_{k>χ} σ_k².
    """
    A, B = tensors[site], tensors[site + 1]
    chi_L = A.shape[0]
    chi_R = B.shape[2]

    # Step 1: Contract
    Theta = np.einsum('aib,bjc->aijc', A, B).reshape(chi_L * 2, 2 * chi_R)

    # Step 2: Apply gate (reshape to expose (i,j) indices)
    G = G4.reshape(2, 2, 2, 2)  # G[i', j', i, j]
    Theta_new = np.einsum('ijkl,akld->aijd', G, Theta.reshape(chi_L, 2, 2, chi_R))
    Theta_new = Theta_new.reshape(chi_L * 2, 2 * chi_R)

    # Step 3: SVD truncation
    U_sv, s, Vh = svd(Theta_new, full_matrices=False)
    chi_new = min(chi_max, len(s))
    if len(s) > 0:
        # Adaptive threshold: drop singular values below 1e-8 × max
        threshold = max(s[0] * 1e-8, 1e-14)
        chi_new = min(chi_new, max(1, int(np.searchsorted(-s, -threshold))))

    # Step 4: Absorb √σ symmetrically
    s_sqrt = np.sqrt(s[:chi_new].astype(np.float32)).astype(DTYPE)
    tensors[site]     = (U_sv[:, :chi_new] * s_sqrt).reshape(chi_L, 2, chi_new)
    tensors[site + 1] = (Vh[:chi_new] * s_sqrt[:, None]).reshape(chi_new, 2, chi_R)


def mps_apply_lr_gate(
    tensors: List[np.ndarray],
    q0: int,
    q1: int,
    gate4: np.ndarray,
    chi_max: int
) -> None:
    """
    Apply a 2-qubit gate between non-neighboring qubits using SWAP chains.

    Moves qubit q0 rightward until adjacent to q1, applies the gate,
    then moves back. Total SWAP cost: 2*(|q1-q0|-1) gates.
    """
    s0, s1 = (q0, q1) if q0 < q1 else (q1, q0)

    for k in range(s0, s1 - 1):
        mps_apply_nn(tensors, SWAP4, k, chi_max)

    mps_apply_nn(tensors, gate4, s1 - 1, chi_max)

    for k in range(s1 - 2, s0 - 1, -1):
        mps_apply_nn(tensors, SWAP4, k, chi_max)


# ─── Gate Fusion ─────────────────────────────────────────────────────────────

def fuse_1q_gates(raw_ops: list) -> list:
    """
    Merge consecutive single-qubit gates on the same qubit into one 2×2 matrix.
    This is exact (no approximation) and can reduce MPS operations by 3-5×.

    Example: RZ(a) → SX → RZ(b) → SX → RZ(c)  →  U_fused (one operation)
    """
    I2 = np.eye(2, dtype=DTYPE)
    pending = {}  # qubit → accumulated matrix (left-to-right product)
    fused = []

    for op in raw_ops:
        if op[0] == '1q':
            q, U = op[1], op[2]
            # New gate applied after existing ones: U_new @ U_accumulated
            pending[q] = U @ pending.get(q, I2)
        else:
            # Two-qubit gate: flush accumulated 1q gates on both qubits first
            q0, q1 = op[1], op[2]
            for q in [q0, q1]:
                if q in pending:
                    fused.append(('1q', q, pending.pop(q)))
            fused.append(op)

    # Flush any remaining pending 1q gates at end of circuit
    for q, U in pending.items():
        fused.append(('1q', q, U))

    return fused


# ─── Decoding ────────────────────────────────────────────────────────────────

def mps_logprob(tensors: List[np.ndarray], bitstring: str) -> float:
    """
    Compute log P(bitstring) by sequential left-to-right contraction.

    Maintains a left boundary vector L of shape (χ_k,):
      L_k+1 = L_k @ T_k[:, bit_k, :]
      logP  += log(|L_k+1|²)
      L_k+1 /= |L_k+1|   (normalize to avoid underflow)

    Cost: O(n · χ²)
    """
    bits = [int(c) for c in bitstring]
    L = np.array([1.0], dtype=DTYPE)
    logp = 0.0

    for site, T in enumerate(tensors):
        v = L @ T[:, bits[site], :]
        prob = float(np.dot(v.conj(), v).real) + 1e-60
        logp += np.log(prob)
        L = v / (np.sqrt(prob) + 1e-30)

    return logp


def beam_search_lr(tensors: List[np.ndarray], W: int = 500) -> List[Tuple]:
    """
    Left-to-right beam search. Returns top-W (logP, bits, env) tuples.

    Output bit ordering: site 0, site 1, ..., site n-1
    (NOT Qiskit qubit ordering — see module docstring)
    """
    n = len(tensors)
    beams = [(0.0, [], np.array([1.0], dtype=DTYPE))]

    for site in range(n):
        T = tensors[site]
        new_beams = []

        for (lp, bits, env) in beams:
            for bit in [0, 1]:
                v = env @ T[:, bit, :]
                prob = float(np.dot(v.conj(), v).real) + 1e-60
                new_beams.append((
                    lp + np.log(prob),
                    bits + [bit],
                    v / (np.sqrt(prob) + 1e-30)
                ))

        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:W]

    return beams


def beam_search_rl(tensors: List[np.ndarray], W: int = 500) -> List[Tuple]:
    """
    Right-to-left beam search. Returns top-W (logP, bits, env) tuples.
    Symmetric to beam_search_lr but contracts from the right boundary.
    """
    n = len(tensors)
    beams = [(0.0, [], np.array([1.0], dtype=DTYPE))]

    for site in range(n - 1, -1, -1):
        T = tensors[site]
        new_beams = []

        for (lp, bits, env) in beams:
            for bit in [0, 1]:
                v = T[:, bit, :] @ env
                prob = float(np.dot(v.conj(), v).real) + 1e-60
                new_beams.append((
                    lp + np.log(prob),
                    [bit] + bits,
                    v / (np.sqrt(prob) + 1e-30)
                ))

        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:W]

    return beams


def bidirectional_decode(
    tensors: List[np.ndarray],
    W: int = 500,
    top_k: int = 50
) -> Tuple[str, str, str, Set[str]]:
    """
    Run beam search from both directions and return the intersection.

    Returns:
        pk_lr   — top-1 bitstring from left-to-right beam
        pk_rl   — top-1 bitstring from right-to-left beam
        best    — highest-scoring bitstring in the intersection (by LR score)
        common  — set of bitstrings appearing in top-k of both directions

    When pk_lr == pk_rl and common is large, confidence is high.
    """
    lr_beams = beam_search_lr(tensors, W)
    rl_beams = beam_search_rl(tensors, W)

    pk_lr = ''.join(map(str, lr_beams[0][1]))
    pk_rl = ''.join(map(str, rl_beams[0][1]))

    top_lr = {''.join(map(str, b[1])) for b in lr_beams[:top_k]}
    top_rl = {''.join(map(str, b[1])) for b in rl_beams[:top_k]}
    common = top_lr & top_rl

    lr_scores = {''.join(map(str, b[1])): b[0] for b in lr_beams}
    if common:
        best = max(common, key=lambda x: lr_scores.get(x, -999))
    else:
        best = pk_lr  # fallback when no intersection

    return pk_lr, pk_rl, best, common


def perturbation_test(
    tensors: List[np.ndarray],
    candidate: str,
    n_rounds: int = 3
) -> Tuple[str, float]:
    """
    Greedy single-bit perturbation. Iteratively accepts any flip that
    improves log-probability until no improving flip exists.

    A candidate that survives this is a strict local maximum under the
    MPS approximation — not a guarantee of global optimality, but a
    necessary condition.

    Returns the (possibly improved) candidate and its log-probability.
    """
    current = candidate
    lp = mps_logprob(tensors, current)
    n = len(current)

    for _ in range(n_rounds):
        improved = False
        for i in range(n):
            bits = list(current)
            bits[i] = str(1 - int(bits[i]))
            mutant = ''.join(bits)
            new_lp = mps_logprob(tensors, mutant)
            if new_lp > lp:
                lp = new_lp
                current = mutant
                improved = True
        if not improved:
            break

    return current, lp


# ─── Utility Functions ────────────────────────────────────────────────────────

def max_bond_dim(tensors: List[np.ndarray]) -> int:
    """Maximum bond dimension across all bonds in the MPS."""
    if len(tensors) <= 1:
        return 1
    return max(T.shape[2] for T in tensors[:-1])


def bond_profile(tensors: List[np.ndarray]) -> List[int]:
    """Bond dimension at each bond position."""
    return [tensors[i].shape[2] for i in range(len(tensors) - 1)]


def mps_norm(tensors: List[np.ndarray]) -> float:
    """
    Compute ⟨ψ|ψ⟩ by contracting the MPS with its conjugate.
    Should be ~1.0 for a normalized state. Useful for sanity checks.
    """
    L = np.ones((1, 1), dtype=DTYPE)
    for T in tensors:
        # L[α',α] → L'[β',β] = Σ_{α',α,i} L[α',α] T*[α',i,β'] T[α,i,β]
        L = np.einsum('ab,aic,bid->cd', L, T.conj(), T)
    return float(abs(L[0, 0]))
