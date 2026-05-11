# TP1: Federated Learning Simulation with Flower

A complete FedAvg implementation using Flower Apps (ClientApp + ServerApp) on FashionMNIST with Dirichlet-partitioned non-IID data.

---

## Project Structure

```
quickstart-pytorch/
├── pyproject.toml              # all hyperparameters + app entry points
├── generate_data.py            # run once before simulation
├── analysis.py                 # ResultsVisualizer for comparing runs
└── pytorchexample/
    ├── __init__.py
    ├── data.py                 # Dirichlet partitioning + data loaders
    ├── model.py                # CustomFashionModel (CNN + train/eval/get/set params)
    ├── client_app.py           # Flower ClientApp (@train, @evaluate)
    └── server_app.py           # Flower ServerApp (manual FedAvg loop)
```

---

## Environment Setup

**Requirements:** Python 3.10+, Ubuntu Linux (or WSL on Windows)

```bash
# 1. Create workspace and virtual environment
mkdir tp1_workspace && cd tp1_workspace
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -U pip
pip install "flwr[simulation]" numpy matplotlib prettytable tomli
pip install torch torchvision

# 3. Generate the Flower app template
flwr new @flwrlabs/quickstart-pytorch
cd quickstart-pytorch

# 4. Replace generated files with TP1 implementations:
#    pytorchexample/data.py       (create new)
#    pytorchexample/model.py      (create new)
#    pytorchexample/client_app.py (replace)
#    pytorchexample/server_app.py (replace)
#    generate_data.py             (create new at project root)
#    analysis.py                  (create new at project root)
```

---

## Configuration

All hyperparameters live in `pyproject.toml` under `[tool.flwr.app.config]`:

```toml
[tool.flwr.app.config]
num-server-rounds = 10
fraction-train = 1.0
fraction-evaluate = 0.5
local-epochs = 1
learning-rate = 0.001
batch-size = 32
num-clients = 20
alpha-dirichlet = 0.5
data-dir = "./data"
seed = 42
client-optimizer = "sgd"      # "sgd" or "adam"
client-algorithm = "fedavg"   # "fedavg" or "fedsgd"
fedprox-mu = 0.0
```

Set the number of simulated clients in `~/.flwr/config.toml`:

```toml
[superlink.local]
options.num-supernodes = 20    # must equal num-clients above
```

> **Rule:** `num-clients` == `num-supernodes` always. Mismatch causes `FileNotFoundError` on missing partition files.

---

## Running a Simulation

### Step 1 — Generate client datasets (required before first run, and after changing `num-clients` or `alpha-dirichlet`)

```bash
python3 generate_data.py
```

This reads `num-clients`, `alpha-dirichlet`, and `data-dir` from `pyproject.toml` and writes `data/client_0.pt ... client_{K-1}.pt`.

### Step 2 — Run the simulation

```bash
flwr run ./                      # runs in background
flwr run ./ --stream             # streams logs to terminal (recommended)
```

Results are saved automatically to `results/<run_id>.json`.

### Step 3 — Analyze results

```bash
python3 analysis.py
```

Figures are saved to `results/figures/`.

---

## Reproducing the Experiments

### Varying alpha (data heterogeneity)

Keep `num-clients=50`, `fraction-train=1.0`. Change `alpha-dirichlet` and rerun:

| Run | `alpha-dirichlet` | Regenerate data? |
|-----|-------------------|-----------------|
| 1   | 0.1               | yes             |
| 2   | 5.0               | yes             |
| 3   | 100.0             | yes             |

```bash
# After each flwr run ./, rename the result file:
mv results/<run_id>.json results/alpha_0_1.json
```

### Varying number of clients

Keep `alpha-dirichlet=0.1`, `fraction-train=1.0`. Change `num-clients` and `num-supernodes` together:

| Run | `num-clients` | `num-supernodes` | Regenerate data? |
|-----|---------------|-----------------|-----------------|
| 1   | 5             | 5               | yes             |
| 2   | 20            | 20              | yes             |
| 3   | 50            | 50              | yes             |

### Varying sampling fraction

Keep `num-clients=50`, `alpha-dirichlet=0.1`. Change only `fraction-train` — **no data regeneration needed**:

| Run | `fraction-train` | Regenerate data? |
|-----|-----------------|-----------------|
| 1   | 0.1             | no              |
| 2   | 0.5             | no              |
| 3   | 1.0             | no              |

### Algorithm comparison (Step 8)

| Run | `client-algorithm` | `client-optimizer` | `num-server-rounds` |
|-----|-------------------|-------------------|---------------------|
| FedAvg + SGD  | `fedavg` | `sgd`  | 10  |
| FedAvg + Adam | `fedavg` | `adam` | 10  |
| FedSGD        | `fedsgd` | `sgd`  | 100 |

> For Adam, use `learning-rate = 0.0001`. For FedSGD, `num-server-rounds = 100` approximates the equivalent number of gradient steps as 10 FedAvg rounds with `local-epochs=1`.

---

## Key Implementation Notes

### When to regenerate data
```
Changed num-clients      → always regenerate
Changed alpha-dirichlet  → always regenerate
Changed fraction-train   → never regenerate
Changed local-epochs     → never regenerate
Changed learning-rate    → never regenerate
```

### Results JSON format

Each `results/<run_id>.json` is a list of round dicts:

```json
[
  {
    "round": 1,
    "num_train_clients": 20,
    "train_loss": 1.42,
    "train_acc": 0.51,
    "fed_eval_loss": 1.87,
    "fed_eval_acc": 0.38,
    "central_loss": 1.65,
    "central_acc": 0.44
  },
  ...
]
```

`fed_eval_*` — weighted average over clients' local val sets (biased by local distribution).  
`central_*` — evaluation on the full balanced FashionMNIST test set (10k samples). Use this for fair cross-run comparison.

### Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: client_14.pt` | `num-supernodes` > `num-clients` | Match both values |
| `Population must be a sequence` | `grid.get_node_ids()` returns a set | Use `sorted(node_ids)` before sampling |
| `'ArrayRecord' has no attribute 'to_numpy_arrays'` | Wrong method name | Use `to_numpy_ndarrays()` |
| High train acc, low eval acc | Local val set shares client's skewed distribution | Use `central_acc` for comparison |

---

## Analysis

Edit the `main()` function in `analysis.py` to point at your result files:

```python
viz = ResultsVisualizer()
viz.add_run("alpha=0.1",  "results/alpha_0_1.json")
viz.add_run("alpha=5.0",  "results/alpha_5_0.json")
viz.add_run("alpha=100",  "results/alpha_100.json")
viz.print_run_summary_table()
viz.plot_all("results/figures/vary_alpha")
```

`plot_all()` saves one PNG per metric: `train_loss`, `train_acc`, `fed_eval_loss`, `fed_eval_acc`, `central_loss`, `central_acc`.
