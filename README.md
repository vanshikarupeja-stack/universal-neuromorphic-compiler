# Universal Neuromorphic Compiler (UNC)

*A hardware-agnostic compiler mapping PyTorch directly to analog spiking silicon.*

---

## Architecture

UNC operates as a multi-pass optimization pipeline designed to decouple standard PyTorch graph definitions from the non-linear, time-dependent constraints of spiking hardware.

 ##  Core Compiler Architecture

UNC utilizes a multi-pass pipeline for mapping PyTorch graphs to spiking hardware.

<pre>
[ PyTorch / snnTorch Graph ]
             │
             ▼  (Pass 1: Graph Tracing & Weight Scaling)
[ Intermediate Quantized Representation ]
             │
             ▼  (Pass 2: Temporal & Synaptic Mapping)
[ Spiking Conductance Graph ]
             │
             ▼  (Pass 3: NIR Serialization)
[ Neuromorphic Target / Colab Analog Simulator ]
</pre>


##  The 3-Pass Compilation Pipeline

### 1. Graph Tracing & Weight Scaling
* **Objective:** Extract the topological layer structure and execution graph from native PyTorch modules.
* **Mechanism:** Converts continuous floating-point synaptic weights into scaled lower-precision representations optimized for surrogate gradient thresholds, reducing memory overhead.

### 2. Temporal Dynamics & Membrane Leak Mapping
* **Objective:** Translate standard continuous spatial activations into discrete temporal spike trains.
* **Mechanism:** Maps Leaky Integrate-and-Fire (LIF) equations across discrete time steps. Calculates membrane voltage decays ($\alpha$) and aligns refractory period limits with target clock cycles.

### 3. Neuromorphic Intermediate Representation (NIR) Serialization
* **Objective:** Generate a standardized, cross-platform intermediate graph representation.
* **Mechanism:** Compiles network topology and LIF neuron parameters into standard `.nir` structures, providing a clean bridge to downstream neuromorphic execution environments.

---

##  Current Status & Roadmap

* **Phase 1 (Live MVP - Current Release):** Zero-install browser verification via Google Colab. Validates end-to-end pipeline ingestion of standard PyTorch models and simulates spiking physics directly.
* **Phase 2 (In Progress):** Full 4-bit dynamic quantization ($W_{\text{int4}}$) and expanded surrogate gradient thresholding.
* **Phase 3 (Roadmap):** Native hardware compilation passes optimized for physical deployment environments (e.g., Intel Loihi 2 / BrainChip Akida architectures).
