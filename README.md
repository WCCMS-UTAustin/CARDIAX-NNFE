# CARDIAX-NNFE

CARDIAX-NNFE is a GPU-accelerated scientific machine learning framework specifically for the Neural Network Finite Element method based on [JAX](https://github.com/google/jax). The major dependencies along with `JAX` are `Equinox` for network creation, `Optax` for optimization procedures, and `CARDIAX` for finite element residual computation. This package is actively managed by the [Willerson Center for Cardiovascular Modeling and Simulation (WCCMS)](https://oden.utexas.edu/research/centers-and-groups/willerson-center-for-cardiovascular-modeling-and-simulation/) and is constantly adapting to accommodate the suite of problems we are intereseted in solving. **We are only focused on GPU development**. The documentation for the software is hosted [here](https://wccms-utaustin.github.io/CARDIAX-NNFE/) and is actively being updated.

![alt text](docs/figures/tutorials/prolate_spheroid/PV_loops.gif)

## Installation

Before installing `nnfe` be sure to install `jax` at [JAX Install](https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-locally-harder). Verify that the GPU is seen by running the following to see if CUDA devices are found

```python
import jax
print(jax.devices())
```

Once the `jax` installation is working, the easiest option is to build all the dependencies through a conda environment using `environment.yaml` which also installs JAX with CUDA. These files set up the pypi dependecies. `CARDIAX` isn't yet pypi, so you can install it through github at [CARDIAX](https://github.com/WCCMS-UTAustin/CARDIAX). Then to install `nnfe`, you must clone and go inside the directory `../NNFE` to run

```
pip install -e .
```


## Examples

In the [documentation](https://wccms-utaustin.github.io/CARDIAX-NNFE/), there are examples that walk through how to use the code. These are under demos, but the files are markdown format to explain functionality. The corresponding `*.py` files live in the `NNFE/demos` directory. The main demo currently is the prolate spheroid, which is the illustrative example in SoftwareX submission.

## Limitations

### Coding
While JAX supports CPU, NNFE is not being tested on CPU environments. We created this codebase to fully leverage GPUs, but the functionality should remain consistent. Also, multi-GPU functionality is also not available. The problems we are currently solving can fit on the memory of a single GPU, so we will not develop this parallelization until needed.

### Finite Element

The scope of finite element limitations are inherited from `CARDIAX`. The rule is if the problem can be solved in traditional FE with `CARDIAX` then you have the ability to attempt to train a network to solve the parameterized problem.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.

## Citations

If you're using this project, you can cite this work [here](CITATION.cff).

We'll add a list of others papers built upon this framework below:

