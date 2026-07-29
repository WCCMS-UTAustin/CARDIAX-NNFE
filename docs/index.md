<div class="wccms-hero" markdown>

<div class="wccms-hero__title">
  <span>CARDIAX-NNFE &nbsp;—&nbsp; developed and maintained by</span>
  <img src="figures/assets/ccs_logo_17.png" alt="WCCMS" style="height: 120px; width: auto; vertical-align: middle; margin-left: 1rem;"/>
</div>

GPU-accelerated Neural Network Finite Element for parameterized cardiac mechanics — a scientific machine learning framework for learning parameter-to-solution maps defined by PDE residuals, built natively on JAX with CARDIAX as the finite element backend.

[Get Started](tutorials/overview.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/WCCMS-UTAustin/CARDIAX-NNFE){ .md-button }

</div>
---

## What is NNFE?

The Neural Network Finite Element (NNFE) method learns the parameter-to-solution mapping of a PDE system directly from its residual. Rather than solving a new boundary value problem for each parameter configuration, NNFE trains a network to approximate the solution — enabling fast, accurate evaluations across a continuous parameterized space.

---

## Key Features

<div class="grid cards" markdown>

-   :material-lightning-bolt: **GPU-Native**

    ---

    Built entirely on JAX. JIT compilation, automatic differentiation, and `vmap`-based batching are first-class citizens throughout the codebase.

-   :material-sigma: **Residual-Based Training**

    ---

    No labeled solution data required. Training is driven entirely by the finite element residual, computed via CARDIAX.

-   :material-tune: **Configuration-Driven**

    ---

    Full experiments are reproducible from YAML configuration files. Each run spawns a unique directory with saved configs, model weights, and plots.

-   :material-graph: **Modular Architecture**

    ---

    Networks, optimizers, samplers, and FE problems are independently configurable. Mix and match components or extend with custom implementations.

</div>

---

## Stack

| Component | Role |
|-----------|------|
| [JAX](https://github.com/google/jax) | Automatic differentiation, JIT compilation, GPU acceleration |
| [CARDIAX](https://github.com/WCCMS-UTAustin/CARDIAX) | Finite element residual computation |
| [Equinox](https://github.com/patrick-kidger/equinox) | Neural network construction and JAX-compatible PyTree handling |
| [Optax](https://github.com/deepmind/optax) | Optimization and learning rate scheduling |
| [Lineax](https://github.com/patrick-kidger/lineax) | GPU-accelerated linear operators |

---

## Installation

Install JAX with CUDA support first — see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder). Then build the full environment:

```bash
conda env create -f environment.yaml
conda activate cardiax
```

Install [CARDIAX](https://github.com/WCCMS-UTAustin/CARDIAX) from GitHub, then install NNFE in editable mode:

```bash
pip install -e .
```

Verify GPU visibility before proceeding:

```python
import jax
print(jax.devices())  # Should show CUDA devices
```

---

## Citation

If you use CARDIAX-NNFE in your research, please cite [`CITATION.cff`](https://github.com/WCCMS-UTAustin/CARDIAX-NNFE/blob/main/CITATION.cff).

*Full citation details will be updated upon paper acceptance in SoftwareX.*