"""
quantum_model.py
────────────────
Stage 3–4: Quantum Feature Encoding + Variational Quantum Circuit (VQC)

Architecture
────────────
  Encoding   :  Amplitude encoding  |ψ⟩ = Σ aᵢ|i⟩  (normalised top-8 CNN features)
  Circuit    :  3 × [ Rx(θ) Rz(φ) on each qubit  +  Ring CNOT entanglement ]
  Measurement:  ⟨Z⟩ expectation values on all 8 qubits → 8-dim quantum output

Implementation
──────────────
  Default  : Pure NumPy statevector simulation (no install needed)
  Optional : PennyLane backend — set USE_PENNYLANE = True
             (pip install pennylane)
  Optional : Qiskit backend  — set USE_QISKIT = True
             (pip install qiskit)

The NumPy simulation is mathematically equivalent to the PennyLane
default.qubit device and produces identical expectation values.
"""

import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────
USE_PENNYLANE = False
USE_QISKIT    = False

N_QUBITS  = 8
N_LAYERS  = 3
N_PARAMS  = N_LAYERS * N_QUBITS * 2   # Rx + Rz per qubit per layer

# Fixed seed weights (simulating trained parameters)
_rng    = np.random.RandomState(42)
VPARAMS = _rng.randn(N_PARAMS).astype(np.float32) * math.pi

if USE_PENNYLANE:
    try:
        import pennylane as qml
        _dev = qml.device("default.qubit", wires=N_QUBITS)
        logger.info("PennyLane backend initialised (default.qubit, %d wires)", N_QUBITS)
    except ImportError:
        USE_PENNYLANE = False
        logger.warning("PennyLane not found — using NumPy statevector simulator")

if USE_QISKIT and not USE_PENNYLANE:
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        logger.info("Qiskit backend initialised")
    except ImportError:
        USE_QISKIT = False
        logger.warning("Qiskit not found — using NumPy statevector simulator")


# ══════════════════════════════════════════════════════════════════
#  PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════

def run_vqc(features_8: np.ndarray) -> np.ndarray:
    """
    Run the 8-qubit VQC and return ⟨Z⟩ expectation values.

    Parameters
    ----------
    features_8 : np.ndarray  float32  shape (8,)  — top-8 CNN features

    Returns
    -------
    measurements : np.ndarray  float64  shape (8,)  — ⟨Zᵢ⟩ ∈ [-1, 1]
    """
    if USE_PENNYLANE:
        return _pennylane_vqc(features_8, VPARAMS)
    if USE_QISKIT:
        return _qiskit_vqc(features_8, VPARAMS)
    return _numpy_vqc(features_8, VPARAMS)


# ══════════════════════════════════════════════════════════════════
#  NumPy Statevector Simulator (default)
# ══════════════════════════════════════════════════════════════════

def _amplitude_encode(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    v = v / (n if n > 1e-8 else 1.0)
    return v / (np.linalg.norm(v) + 1e-12)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[math.e ** (-0.5j * theta), 0],
         [0,                         math.e ** (0.5j * theta)]],
        dtype=complex
    )


def _apply_gate_1q(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    new = np.zeros(2 ** n, dtype=complex)
    for i in range(2 ** n):
        b = (i >> (n - 1 - qubit)) & 1
        for ob in range(2):
            new[i ^ ((b ^ ob) << (n - 1 - qubit))] += gate[ob, b] * state[i]
    return new


def _apply_cnot(state: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
    new = state.copy()
    for i in range(2 ** n):
        if (i >> (n - 1 - ctrl)) & 1:
            j = i ^ (1 << (n - 1 - tgt))
            new[i], new[j] = state[j], state[i]
    return new


def _numpy_vqc(f8: np.ndarray, params: np.ndarray) -> np.ndarray:
    N = N_QUBITS
    amps = _amplitude_encode(f8)

    # Initialise statevector
    state = np.zeros(2 ** N, dtype=complex)
    for i in range(N):
        state[1 << (N - 1 - i)] = amps[i] if i < len(amps) else 0.0
    nm = np.linalg.norm(state)
    state = state / nm if nm > 1e-8 else np.eye(2 ** N)[0] + 0j

    # Variational layers
    pi = 0
    for _ in range(N_LAYERS):
        for q in range(N):
            if pi < len(params):
                state = _apply_gate_1q(state, _rx(params[pi]), q, N); pi += 1
            if pi < len(params):
                state = _apply_gate_1q(state, _rz(params[pi]), q, N); pi += 1
        # Ring CNOT entanglement
        for q in range(N):
            state = _apply_cnot(state, q, (q + 1) % N, N)

    # ⟨Z⟩ measurement
    return np.array([
        sum((1 - 2 * ((i >> (N - 1 - q)) & 1)) * abs(state[i]) ** 2
            for i in range(2 ** N))
        for q in range(N)
    ], dtype=float)


# ══════════════════════════════════════════════════════════════════
#  PennyLane backend (optional)
# ══════════════════════════════════════════════════════════════════

def _pennylane_vqc(f8: np.ndarray, params: np.ndarray) -> np.ndarray:
    import pennylane as qml

    amps = _amplitude_encode(f8)
    # Pad to 2^N
    state_init = np.zeros(2 ** N_QUBITS, dtype=complex)
    for i in range(N_QUBITS):
        state_init[1 << (N_QUBITS - 1 - i)] = amps[i] if i < len(amps) else 0.0
    nm = np.linalg.norm(state_init)
    if nm > 1e-8:
        state_init /= nm

    @qml.qnode(_dev)
    def circuit():
        qml.QubitStateVector(state_init, wires=range(N_QUBITS))
        pi = 0
        for _ in range(N_LAYERS):
            for q in range(N_QUBITS):
                if pi < len(params): qml.RX(params[pi], wires=q); pi += 1
                if pi < len(params): qml.RZ(params[pi], wires=q); pi += 1
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

    return np.array(circuit(), dtype=float)


# ══════════════════════════════════════════════════════════════════
#  Qiskit backend (optional)
# ══════════════════════════════════════════════════════════════════

def _qiskit_vqc(f8: np.ndarray, params: np.ndarray) -> np.ndarray:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    amps = _amplitude_encode(f8)
    init = np.zeros(2 ** N_QUBITS, dtype=complex)
    for i in range(N_QUBITS):
        init[1 << (N_QUBITS - 1 - i)] = amps[i] if i < len(amps) else 0.0
    nm = np.linalg.norm(init)
    if nm > 1e-8: init /= nm

    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(init.tolist(), range(N_QUBITS))
    pi = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            if pi < len(params): qc.rx(float(params[pi]), q); pi += 1
            if pi < len(params): qc.rz(float(params[pi]), q); pi += 1
        for q in range(N_QUBITS):
            qc.cx(q, (q + 1) % N_QUBITS)

    sv = Statevector(qc)
    probs = np.abs(sv.data) ** 2
    measurements = np.array([
        sum((1 - 2 * ((i >> (N_QUBITS - 1 - q)) & 1)) * probs[i]
            for i in range(2 ** N_QUBITS))
        for q in range(N_QUBITS)
    ], dtype=float)
    return measurements
