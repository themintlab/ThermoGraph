# ThermoGraph

[![Unit Tests](https://github.com/themintlab/ThermoGraph/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/themintlab/ThermoGraph/actions/workflows/unit-tests.yml)

**A domain compiler and application layer bridging classical thermodynamics with modern tensor networks.**

`thermograph` is the user-facing thermodynamic application layer built on top of [**ZGraph**](https://github.com/themintlab/ZGraph). It acts as a domain compiler—translating human-readable physical concepts (elements, phases, CALPHAD polynomials, microstate ensembles, and state variables like $T, P, \mu$) into the strict, unlabelled tensor graphs required by the `zgraph` execution engine.

---

## 🏛️ Core Philosophy: Knows Physics, Delegates Math

`thermograph` **knows physics, but does zero math directly**.

All mathematical evaluation, tensor contraction, soft-minimum selection, and auto-differentiation are delegated to the underlying [`zgraph`](https://github.com/themintlab/ZGraph) engine. `thermograph` handles model definition, variable index packing, unit conversions, and physical envelope management.

### Division of Responsibilities

| Feature / Responsibility | **`thermograph`** (Application Layer) | **`zgraph`** (Math Engine) |
| :--- | :--- | :--- |
| **Primary Focus** | Domain semantics, CALPHAD models, phase equilibrium | Soft-tropical tensor algebra, kernel execution |
| **Knowledge** | **Knows physics**, elements, phases, units | **Physics-free**, pure tensor math |
| **Data Representation** | Human-readable named objects, DataFrames, signals | Unlabeled JAX Arrays (`jax.Array`) |
| **Execution Role** | Graph compilation, index linker, envelope swapping | Tensor contraction, `jax.jit` kernels |
| **Introspection** | Offline inspection (`np.array()`), plots | High-throughput autodiff & Hessian calculation |

---

## ⚡ Key Architecture & Features

### 1. The Linker & Compiler Pattern
`thermograph` maps complex thermodynamic systems (such as SGTE unary databases, Redlich-Kister interaction polynomials, or sublattice models) into structured input channel signal vectors. It compiles physical equations into lightweight `zgraph` factor and leaf nodes.

### 2. Envelope Replacement (Safe Physics Hot-Swapping)
Because `zgraph` tensor execution graphs are strictly immutable to maintain `jax.jit` optimization, `thermograph` handles physical model changes (e.g., changing from an ideal solution to a sub-regular solution model) by rebuilding and replacing the lightweight `zgraph` node envelopes without compromising kernel stability.

### 3. Offline Introspection
All user-facing diagnostic tools and DataFrame exports (e.g., microstate configuration matrices, phase fraction summaries) safely convert from JAX arrays via `np.array()`, enabling rich interactive analysis without breaking autodiff.

---

## 💻 Application Usage Example

Using `thermograph` and `zgraph` to construct and evaluate thermodynamic phase models and extract phase boundaries:

```python
import jax.numpy as jnp
import zgraph as zg, thermograph as tg

# 1. Define coordinate signals (T, mu_Cu, mu_Ni)
T, mu_Cu, mu_Ni = zg.SignalNodes(0, 1, 2)
RT = tg.KB_R * T

# 2. Load pre-compiled SGTE unary reference models
cu_fcc = tg.load_library("unary/GHSERCU")
ni_fcc = tg.load_library("unary/GHSERNI")
cu_liq = tg.load_library("unary/GLIQCU")
ni_liq = tg.load_library("unary/GLIQNI")

# 3. Assemble ideal solution phases and root equilibrium system
phase_FCC = zg.FactorNode(jnp.eye(2), [mu_Cu - cu_fcc, mu_Ni - ni_fcc], tau=RT)
phase_LIQ = zg.FactorNode(jnp.eye(2), [mu_Cu - cu_liq, mu_Ni - ni_liq], tau=RT)
system = zg.FactorNode(jnp.eye(2), [phase_FCC, phase_LIQ], tau=0.0)

# 4. Evaluate grand potential or extract T-x phase boundaries
signals = jnp.array([1500.0, -80000.0, -80000.0])
omega = system(signals)
```

---

## 🚀 Interactive Examples

Explore interactive Jupyter notebooks running natively on Google Colab:

- **Building the SGTE Unary Library** (`build_sgte_library.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/build_sgte_library.ipynb)
- **Cu-Ni Phase Diagram** (`cu_ni_phase_diagram.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/cu_ni_phase_diagram.ipynb)
- **Cu-Au Phase Diagram** (`cu_au_phase_diagram.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/cu_au_phase_diagram.ipynb)
- **Si-Ge Phase Diagram** (`si_ge_phase_diagram.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/si_ge_phase_diagram.ipynb)
- **Third-Generation CALPHAD: Einstein Oscillator** (`einstein_oscillator.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/einstein_oscillator.ipynb)
- **Au-Ni Solid Miscibility Gap** (`au_ni_miscibility.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/au_ni_miscibility.ipynb)
- **Cu-Mg Intermetallic Line Compounds** (`cu_mg_intermetallics.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/cu_mg_intermetallics.ipynb)
- **Bayesian Training & Uncertainty Propagation** (`bayesian_si_training.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ThermoGraph/blob/main/examples/bayesian_si_training.ipynb)

---

## 🔗 Underlying Math Engine

For low-level tensor kernel operations, soft-tropical semirings, and custom differentiable graph construction, see the [**ZGraph**](https://github.com/themintlab/ZGraph) engine repository.

