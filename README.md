# TP: Federated Learning Simulation with Flower

A complete FedAvg implementation using Flower Apps (ClientApp + ServerApp) on FashionMNIST with Dirichlet-partitioned non-IID data.
Implementation of FedAvg, FedProx, SCAFFOLD, and Byzantine-robust aggregation (FedMedian, Krum) using Flower Apps on FashionMNIST with Dirichlet-partitioned non-IID data.

---

## Project Structure

```
quickstart-pytorch/
├── pyproject.toml              # all hyperparameters + app entry points
├── generate_data.py            # run once before simulation
├── analysis.py                 # ResultsVisualizer for comparing runs
├── data/client_n.pt            # client split data using dirichlet distribution
└── pytorchexample/
    ├── __init__.py
    ├── data.py                 # Dirichlet partitioning + data loaders
    ├── model.py                # CustomFashionModel (CNN + train/eval/get/set params)
    ├── client_app.py           # Flower ClientApp (@train, @evaluate)
    └── server_app.py           # Flower ServerApp (manual FedAvg loop)
├── results
    ├── tp1_results
    ├── tp2_results
    ├── tp3_results
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

```

---

## Configuration

All hyperparameters live in `pyproject.toml` under `[tool.flwr.app.config]`:

```toml
[tool.flwr.app.config]
## TP1 Parameters
num-server-rounds = 20
fraction-train = 1
fraction-evaluate = 0.5
local-epochs = 3
learning-rate = 0.01
batch-size = 64
num-clients = 20
alpha-dirichlet = 0.01  # alpha = control partition and IID
data-dir = "./data"
seed = 42

## TP2 Parameters: algorithm and optimizer
client-optimizer = "sgd" # "sgd" or "adam"
client-algorithm = "fedavg" ## "fedavg" or "fedsgd" or "scaffold"
fedprox-mu = 0

## TP3 Parameters: poison type and strategy
malicious-ratio = 0 # 0 = no attack
attack-type = "model"  ## "data" or "model"
strategy = "fedavg" ## "fedavg", "fedmedian", or "krum"
krum-f = 0  # num malicious clients
output-dir = "./results/tp3_results/heterogeneity" ## Output directory
```

Set the number of simulated clients in `~/.flwr/config.toml`:

```toml
[superlink.local]
options.num-supernodes = 20    # must equal num-clients above
```

> **Rule:** `num-clients` == `num-supernodes` always. Mismatch causes `Error`.

---

## Running a Simulation

### Step 1 — Generate client datasets (required before first run, and after changing `num-clients` or `alpha-dirichlet`)

```bash
python generate_data.py
```

This reads `num-clients`, `alpha-dirichlet`, and `data-dir` from `pyproject.toml` and writes `data/client_0.pt ... client_{K-1}.pt`.

### Step 2 — Run the simulation

```bash
flwr run ./                      # runs in background
flwr run ./ --stream             # streams logs to terminal
```

Results are saved automatically to specified `output-dir` in config as json.

### Step 3 — Analyze results

```bash
python analysis.py
```

Figures are saved to the specified directory.
```python
viz.plot_all("results/figures/vary_alpha")
```

`plot_all()` saves one PNG per metric: `train_loss`, `train_acc`, `fed_eval_loss`, `fed_eval_acc`, `central_loss`, `central_acc`.

---
## TP1: FedAvg — Effect of Heterogeneity, Client Count, and Sampling
### Reproducing the Experiments -- TP1

### Varying alpha (data heterogeneity)

Keep `num-clients=50`, `fraction-train=1.0`. Change `alpha-dirichlet` and rerun:

| Run | `alpha-dirichlet` | Expected trend                                |
|-----|-------------------|-----------------------------------------------|
| 1   | 0.1               | Fast local convergence, high local-global gap |
| 2   | 5.0               | Moderate convergence                          |
| 3   | 100.0             | Slower but more stable, lower gap             |

```bash
# After each flwr run ./, rename the result file:
mv results/<run_id>.json results/alpha_0_1.json
```

### Varying number of clients

Keep `alpha-dirichlet=0.1`, `fraction-train=1.0`. Change `num-clients` and `num-supernodes` together:

| Run | `num-clients` | `num-supernodes`|
|-----|---------------|-----------------|
| 1   | 5             | 5               |
| 2   | 20            | 20              |
| 3   | 50            | 50              |

### Varying sampling fraction

Keep `num-clients=50`, `alpha-dirichlet=0.1`. Change only `fraction-train`:

| Run | `fraction-train`| Expected Behavior                 |
|-----|-----------------|-----------------------------------|
| 1   | 0.1             | Unstable, near-random convergence |
| 2   | 0.5             | Moderate stability                |
| 3   | 1.0             | Smoothest, fastest convergence    |

### Varying local epochs

Keep `num-clients=50`, `alpha-dirichlet=0.1`. Change only `fraction-train`:

| Run | `fraction-train`|
|-----|-----------------|
| 1   | 1               |
| 2   | 5               |

### Algorithm comparison (Step 8)

| Run           | `client-algorithm-Optimizer` |
|---------------|------------------------------|
| FedAvg + SGD  | `fedavg`       |     `sgd`   | 
| FedAvg + Adam | `fedavg`       |     `adam`  |
| FedSGD        | `fedsgd`       |     `sgd`   |

> For Adam, use `learning-rate = 0.001`. For FedSGD, `num-server-rounds` needs to high as it converges very slowly.

---

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

`train_acc_*` — local training accuracy.  
`fed_eval_*` — weighted average over clients' local val sets (biased by local distribution).  
`central_*` — evaluation on the full balanced FashionMNIST test set (10k samples). Used for fair cross-run comparison.

---

## TP2: Data Heterogeneity — FedProx and SCAFFOLD
### Reproducing the Experiments -- TP2

TP2 uses the same codebase with additional algorithms. Default config for all TP2 runs:

```toml
num-server-rounds = 10 or 20
num-clients = 20 or 50
fraction-train = 1.0
local-epochs = 3
learning-rate = 0.01
batch-size = 64
```

Update `num-supernodes` accordingly in `~/.flwr/config.toml`.

### Algorithms

**FedAvg** — baseline, no changes needed.

**FedProx** — add proximal term to client loss:
```toml
fedprox-mu = 0.1       # proximal coefficient
client-algorithm = "fedavg"
```

**SCAFFOLD** — control variate correction:
```toml
client-algorithm = "scaffold"
fedprox-mu = 0.0
```

### Run Sequence (for each algorithm)

Set the alpha value and generate the dataset. Run FedProx (`fedprox-mu = 0.1`) and SCAFFOLD (`client-algorithm = "scaffold"`). Vary the alpha value to test the impact of heterogeniety on utility of fedprox and scaffold.

---

## TP3: Security — Attacks and Robust Aggregation
### Reproducing the Experiments -- TP3

Use: `num-clients=20`, `num-supernodes=20`, `alpha-dirichlet=10.0` (IID-ish for attack experiments), `num-server-rounds=20`, `local-epochs=3`, `lr=0.01`

### Experiment 1 — FedAvg Under Attack
Fixed: `strategy=fedavg`

| Run | `malicious-ratio`| `attack-type` |
|-----|------------------|---------------|
| 1   | 0.0              | data          |
| 2   | 0.25             | data          |
| 3   | 0.50             | data          |
| 4   | 0.25             | model         |
| 5   | 0.50             | model         |

> For model poisoning use `attack-scale=2.0` in `client_app.py` to avoid NaN explosion at 50%.

### Experiment 2 — FedMedian Defense
Change `strategy=fedmedian`. Run all five attack scenarios similar to baseline attack without strategies.

### Experiment 3 — Krum Defense
Change `strategy=krum`. Set `krum-f` based on expected malicious count:

| `malicious-ratio`| `num-clients`| `krum-f`| Note                            |
|------------------|--------------|---------|---------------------------------|
| 0.0              | 20           | 0       | No filtering                    |
| 0.25             | 20           | 5       | 25% of 50                       |
| 0.50             | 20           | 8       | Max safe f for n=20: (20-3)/2=8 |

### Experiment 4 — Heterogeneity Impact on Defenses
Fixed: `malicious-ratio=0.25`, `attack-type=model`, `attack-scale=2.0`, `num-clients=20`

Vary `alpha-dirichlet ∈ {10, 0.1}` across all three strategies. Regenerate data when alpha changes.

| Run | Alpha | Strategy  | `krum-f |
|-----|-------|-----------|---------|
| 1   | 10    | fedavg    | 0       |
| 3   | 0.1   | fedavg    | 0       |
| 4   | 10    | fedmedian | 0       |
| 6   | 0.1   | fedmedian | 0       |
| 7   | 10    | krum      | 5       |
| 9   | 0.1   | krum      | 5       |

---

## Key Implementation Notes

**Metric reliability:**
- `train_acc` — biased by local distribution; high for non-IID does not mean good global model
- `fed_eval_acc` — biased by local val distribution; unreliable for cross-run comparison
- `central_acc` — ground truth; evaluated on balanced 10k FashionMNIST test set; use this

**Malicious client assignment (TP3):**
Clients with `partition_id < floor(malicious_ratio × num_clients)` are malicious — deterministic, stable across rounds.

**Krum theoretical limit:**
Valid only when `n ≥ 2f + 3`. For `n=50`, maximum `f=23`. Setting `f` too high degrades performance even without attack.

**FedMedian guarantee:**
Coordinate-wise median is robust only when fewer than 50% of clients are malicious. At exactly 50%, the guarantee breaks and performance may be worse than FedAvg.

---

## Conclusion

### TP1 — FedAvg Foundations

FedAvg is a practical and effective algorithm, but its performance is largely affected by three factors.

**Data heterogeneity (alpha)** is the dominant factor. Low alpha creates client drift, local models specialize on their dominant classes and produce conflicting gradients. The result is a large gap between local training accuracy (high) and global test accuracy (low). High alpha produces more honest updates but converges more slowly per round because each update is less decisive.

**Number of clients** primarily affects convergence speed and stability, not final accuracy. Fewer clients means more data per client and faster early convergence. More clients increases round-to-round oscillation under non-IID data because the sampled subset composition varies more. Final accuracy under sufficient rounds converges to similar values regardless of K, heterogeneity is the bottleneck, not client count.

**Sampling fraction** is the most operationally critical parameter. At f=0.1 with 50 clients and alpha=0.1, only 5 clients participate per round. It acts as trade-off between number of communication cost and stable learning. With higher ratio the model generalizes better but has high communication costs.

**Algorithm comparison** shows FedAvg+Adam converges faster in early rounds but reaches similar final accuracy to FedAvg+SGD. FedSGD requires many more rounds to match FedAvg because a single batch step per round accumulates knowledge slowly. FedAvg + Adam performs better then FedAvg + SGD as the data becomes more IID, as large gradient shifts due to momemtum leads to client drift for non-IID data. 

### TP2 — Heterogeneity and Client Drift

The fundamental problem identified in TP1 — client drift under non-IID data — has a precise mechanism and two targeted algorithmic solutions.

**Client drift** occurs because FedAvg's local optimization has no constraint on how far clients can move from the global model. Under high heterogeneity, multiple local epochs compound this divergence, each step moves clients further toward their local optimum and away from the global objective.

**FedProx** addresses drift passively by adding a proximal term that penalizes large deviations from the global model during local training. It behaves identically to FedAvg when heterogeneity is low (the penalty term is near zero when honest updates agree) and provides increasing benefit as alpha decreases. The mu hyperparameter requires tuning as too small and the effect is negligible, too large and clients cannot learn their local data effectively.

**SCAFFOLD** addresses drift actively by maintaining control variates — estimates of each client's gradient bias relative to the global gradient. By correcting the gradient direction during local training, SCAFFOLD enables faster, more stable convergence under extreme heterogeneity. Under high non-IID (alpha=0.01), SCAFFOLD outperforms both FedAvg and FedProx in convergence speed and final accuracy. The cost is additional communication per round (control variates must be exchanged) and increased implementation complexity.

**The core TP2 finding:** When data heterogeneity is low, all three algorithms perform similarly and FedAvg is the pragmatic choice. As heterogeneity increases, FedProx provides incremental improvement and SCAFFOLD provides substantial improvement but the appropriate choice depends on the communication budget and deployment constraints.

**Central evaluation** is essential for fair comparison across algorithms. Federated evaluation on local validation sets is biased by local data distribution and gives misleading signals appearing artificially high for non-IID settings and near-random for IID settings.

### TP3 — Security and Robustness

FL's decentralized nature is both its privacy strength and its security weakness. The server cannot observe local training, making it impossible to verify whether clients train honestly.

**Data poisoning** (label flipping) is a persistent but moderate threat under FedAvg. At 25% malicious ratio it slows convergence without preventing learning. At 50% clients create opposing gradients that keeps the model oscillating. Data poisoning requires majority control to fully corrupt FedAvg because the attack works against honest gradient competition.

**Model poisoning** (weight scaling with sign reversal) is catastrophically more effective. At 25% malicious ratio with scale=2, FedAvg is completely destroyed after several rounds. The mechanism is geometric: negative scaling repeatedly undoes the previous round's learning, creating a permanent equilibrium at the initialization point. Scale factor, malicious ratio, and FedAvg's linear averaging combine to make recovery mathematically impossible without changing the aggregation rule.

**FedMedian** provides strong protection against both attacks when fewer clients are malicious. However, FedMedian fails at exactly 50% poisoning and at the boundary, the median falls in the corrupted zone. FedMedian is the practical default defense: it requires no knowledge of f, adds minimal overhead under honest conditions, and handles the most dangerous class of attacks.

**Krum** provides strong protection against model poisoning when the malicious count is correctly estimated, achieving accuracy comparable to FedMedian. Its critical weakness is that it selects a single client update discarding all other information and can lead to slower convergence. Against data poisoning, Krum fails as the malicious clients grow because poisoned clients form a coherent cluster that Krum cannot distinguish from the honest cluster. Krum is appropriate only when the attack type is known to be magnitude-based.

**The heterogeneity-robustness tension** is TP3's most important insight. Under IID data, honest clients form a tight cluster and robust defenses work cleanly. Under high heterogeneity (alpha=0.1), honest clients naturally look different from each other — their updates point in different directions due to different local data distributions. This makes them resemble malicious clients to both FedMedian (their values are spread, pushing the median away from any single honest update) and Krum (honest clients have high inter-client distances, similar to malicious outliers). The defense that protects against Byzantine attacks under IID conditions progressively misidentifies honest non-IID clients as suspicious under increasing heterogeneity. There is no defense that simultaneously handles both high heterogeneity and high malicious ratio without tradeoffs, hence this remains an open research problem in federated learning security.