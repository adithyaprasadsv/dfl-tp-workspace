"""pytorchexample: A Flower / PyTorch app."""

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from pytorchexample.model import CustomFashionModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

## sample clients
def sample_clients(node_ids, fraction: float, global_rng: random.Random) -> list:
    node_ids_list = sorted(node_ids) 
    k = max(1, int(len(node_ids_list) * fraction))
    return global_rng.sample(node_ids_list, k)

## FedAvg aggregation
def fedavg(state_dicts: List[Dict[str, torch.Tensor]], num_examples: List[int]) -> Dict[str, torch.Tensor]:
    total = sum(num_examples)
    aggregated = {}

    for key in state_dicts[0].keys():
        # Stack weighted tensors and sum
        aggregated[key] = torch.stack([
            state_dicts[i][key].float() * (num_examples[i] / total)
            for i in range(len(state_dicts))
        ]).sum(dim=0)

    return aggregated

## Metric aggregation
def aggregate_metrics(metrics: List[Dict[str, float]], num_examples: List[int]) -> Dict[str, float]:
    total = sum(num_examples)
    aggregated = {}

    for key in metrics[0].keys():
        aggregated[key] = sum(
            metrics[i][key] * (num_examples[i] / total)
            for i in range(len(metrics))
        )

    return aggregated

# a helper function to convert ArrayRecord to state_dict
def arrays_to_state_dict(arrays: ArrayRecord, model: CustomFashionModel) -> Dict[str, torch.Tensor]:
    keys = list(model.state_dict().keys())
    params = arrays.to_numpy_ndarrays()  
    return {
        key: torch.tensor(val)
        for key, val in zip(keys, params)
    }

def list_to_array_record(arrays: list) -> ArrayRecord:
    return ArrayRecord(arrays)

def array_record_to_list(record: ArrayRecord) -> list:
    return record.to_numpy_ndarrays()

# Create ServerApp
# app = ServerApp()

# @app.main()
# def main(grid: Grid, context: Context) -> None:
#     """Main entry point for the ServerApp."""

#     # Read run config
#     fraction_evaluate: float = context.run_config["fraction-evaluate"]
#     num_rounds: int = context.run_config["num-server-rounds"]
#     lr: float = context.run_config["learning-rate"]

#     # Load global model
#     global_model = Net()
#     arrays = ArrayRecord(global_model.state_dict())

#     # Initialize FedAvg strategy
#     strategy = FedAvg(fraction_evaluate=fraction_evaluate)

#     # Start strategy, run FedAvg for `num_rounds`
#     result = strategy.start(
#         grid=grid,
#         initial_arrays=arrays,
#         train_config=ConfigRecord({"lr": lr}),
#         num_rounds=num_rounds,
#         evaluate_fn=global_evaluate,
#     )

#     # Save final model to disk
#     print("\nSaving final model to disk...")
#     state_dict = result.arrays.to_torch_state_dict()
#     torch.save(state_dict, "final_model.pt")


# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # hyperparameters from pyproject.toml
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])
    fraction_eval = float(context.run_config["fraction-evaluate"])
    lr = float(context.run_config["learning-rate"])
    local_epochs = int(context.run_config["local-epochs"])
    seed = int(context.run_config.get("seed", 42))
    run_id = str(context.run_id)
    mu = float(context.run_config.get("fedprox-mu", 0.0))

    global_rng = random.Random(seed)

    # global model initialization
    global_model = CustomFashionModel()
    initial_arrays = ArrayRecord(global_model.get_model_parameters())

    algorithm = str(context.run_config.get("client-algorithm", "fedavg"))
    # Initialize global control variate c to zeros (same shape as model params)
    if algorithm == "scaffold":
        print("initializing global control variate c to zeros")
        c_global = [np.zeros_like(p) for p in global_model.get_model_parameters()]
    else:
        c_global = None

    # client node IDs
    all_node_ids = grid.get_node_ids()   # returns set[int]
    print(f"\nAvailable clients: {len(all_node_ids)}")

    results = []

    # FL ROUNDS
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound: {round_num}/{num_rounds}")
        current_arrays = ArrayRecord(global_model.get_model_parameters())

        train_clients = sample_clients(all_node_ids, fraction_train, global_rng)
        print(f" Training clients: {len(train_clients)}")

        from flwr.app import Message, RecordDict
        # train_messages = [
        #     Message(
        #         content=RecordDict({
        #             "arrays": current_arrays,
        #             "config": ConfigRecord({"lr": lr, "local-epochs": local_epochs, "mu": mu}),
        #         }),
        #         message_type="train",
        #         dst_node_id=nid,
        #         group_id=str(round_num),
        #     )
        #     for nid in train_clients
        # ]

        if algorithm == "scaffold":
            msg_content = RecordDict({
                "arrays":   current_arrays,
                "c_global": list_to_array_record(c_global),
                "config":   ConfigRecord({"lr": lr, "local-epochs": local_epochs, "mu": 0.0}),
            })
        else:
            msg_content = RecordDict({
                "arrays": current_arrays,
                "config": ConfigRecord({"lr": lr, "local-epochs": local_epochs, "mu": mu}),
            })

        train_messages = [
            Message(
                content=msg_content,
                message_type="train",
                dst_node_id=nid,
                group_id=str(round_num),
            )
            for nid in train_clients
        ]

        train_replies = list(grid.send_and_receive(train_messages))

        client_state_dicts = []
        client_num_examples = []
        client_train_metrics = []
        client_c_locals = []   # for scaffold

        for reply in train_replies:
            if not reply.has_content():
                print(f"  WARNING: train reply has no content, skipping.")
                continue

            sd = arrays_to_state_dict(reply.content["arrays"], global_model)
            client_state_dicts.append(sd)

            m = dict(reply.content["metrics"])
            client_num_examples.append(int(m.pop("num-examples")))
            client_train_metrics.append(m)

            if algorithm == "scaffold":
                ck = array_record_to_list(reply.content["c_local"])
                client_c_locals.append(ck)

        if not client_state_dicts:
            print(f"  Round {round_num}: no valid replies, skipping.")
            results.append({
                "round": round_num, "num_train_clients": 0,
                "train_loss": None, "train_acc": None,
                "fed_eval_loss": None, "fed_eval_acc": None,
                "central_loss": None, "central_acc": None,
            })
            continue

        if algorithm == "scaffold" and client_c_locals:
            n = len(client_c_locals)
            c_global = [
                c_i + (1.0 / n) * sum(ck[i] - c_i for ck in client_c_locals)
                for i, c_i in enumerate(c_global)
            ]

        # Aggregate & update global model
        strategy = str(context.run_config.get("strategy", "fedavg"))
        krum_f = int(context.run_config.get("krum-f", 0))

        # Inside the round loop, replace the fedavg call:
        if strategy == "fedmedian":
            print("Performing FedMedian aggregation")
            new_state_dict = fedmedian(client_state_dicts)
        elif strategy == "krum":
            print(f"Performing Krum aggregation with f={krum_f}")
            new_state_dict = krum(client_state_dicts, f=krum_f)
        else:
            # default: fedavg
            new_state_dict = fedavg(client_state_dicts, client_num_examples)

        global_model.load_state_dict(new_state_dict)

        # new_state_dict = fedavg(client_state_dicts, client_num_examples)
        # global_model.load_state_dict(new_state_dict)

        agg_train = aggregate_metrics(client_train_metrics, client_num_examples)
        print(f"  Train loss: {agg_train.get('train_loss', 0):.4f} | "
              f"Train acc: {agg_train.get('train_acc', 0):.4f}")

        # EVALUATION 
        eval_clients = sample_clients(all_node_ids, fraction_eval, global_rng)
        updated_arrays = ArrayRecord(global_model.get_model_parameters())

        eval_messages = [
            Message(
                content=RecordDict({"arrays": updated_arrays}),
                message_type="evaluate",
                dst_node_id=nid,
                group_id=str(round_num),
            )
            for nid in eval_clients
        ]

        eval_replies = list(grid.send_and_receive(eval_messages))

        client_eval_metrics = []
        client_eval_examples = []

        for reply in eval_replies:
            if not reply.has_content():
                print(f"  WARNING: evaluate reply has no content, skipping.")
                continue
            m = dict(reply.content["metrics"])
            client_eval_examples.append(int(m.pop("num-examples")))
            client_eval_metrics.append(m)
        
        if not client_eval_metrics:
            agg_eval = {"eval_loss": None, "eval_acc": None}
        else:
            agg_eval = aggregate_metrics(client_eval_metrics, client_eval_examples)
        
        print(f"  Eval  loss: {agg_eval.get('eval_loss', 0):.4f} | "
              f"Eval  acc: {agg_eval.get('eval_acc', 0):.4f}")
        
        # Store results
        # results.append({
        #     "round": round_num,
        #     "num_train_clients": len(train_clients),
        #     "train_loss": agg_train.get("train_loss", None),
        #     "train_acc": agg_train.get("train_acc", None),
        #     "fed_eval_loss": agg_eval.get("eval_loss", None),
        #     "fed_eval_acc": agg_eval.get("eval_acc", None),
        # })

        # After fedavg aggregation, add:
        central_loss, central_acc = global_evaluate(global_model)
        print(f"  Central test loss: {central_loss:.4f} | Central test acc: {central_acc:.4f} ")

        # Store results
        results.append({
            "round": round_num,
            "num_train_clients": len(train_clients),
            "train_loss": agg_train.get("train_loss", None),
            "train_acc": agg_train.get("train_acc", None),
            "fed_eval_loss": agg_eval.get("eval_loss", None),
            "fed_eval_acc": agg_eval.get("eval_acc", None),
            "central_loss": central_loss,
            "central_acc": central_acc
        })

    # save as json
    output_dir = str(context.run_config.get("output-dir", "./results/tp3_results"))
    results_dir = Path(output_dir)
    # results_dir = Path("results/vary_alpha") ## path to save results for different experiments
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{run_id}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {out_path}")

    # Save final model weights
    torch.save(global_model.state_dict(), "final_model.pt")
    print("Final model saved")

def global_evaluate(model: CustomFashionModel) -> tuple[float, float]:
    """Evaluate model on central data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(
        root="./raw_data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loss, acc = model.test_epoch(test_loader, criterion, device)
    return loss, acc

# counter posisoning attack
def fedmedian(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    aggregated = {}
    for key in state_dicts[0].keys():
        # Stack all clients' tensors for this parameter
        # Shape: (num_clients, *param_shape)
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)

        # Coordinate-wise median along client dimension (dim=0)
        aggregated[key] = torch.median(stacked, dim=0).values

    return aggregated

def krum(state_dicts: List[Dict[str, torch.Tensor]], f: int) -> Dict[str, torch.Tensor]:
    n = len(state_dicts)
    k = n - f - 2
    if k < 1:
        k = 1

    # 1. Flatten 
    flat_params = []
    for sd in state_dicts:
        flat = np.concatenate([
            val.cpu().numpy().flatten()
            for val in sd.values()
        ])
        flat_params.append(flat)

    flat_params = np.array(flat_params)

    # 2. Compute Euclidean distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = np.linalg.norm(flat_params[i] - flat_params[j])

    # 3. Score 
    scores = np.zeros(n)
    for i in range(n):
        neighbor_distances = sorted(
            [distances[i][j] for j in range(n) if j != i]
        )
        scores[i] = sum(neighbor_distances[:k])

    # 4. Select client with lowest score
    best_client = int(np.argmin(scores))
    print(f"  Krum selected client {best_client} (score={scores[best_client]:.4f})")

    return state_dicts[best_client]

# def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
#     """Evaluate model on central data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(arrays.to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load entire test set
#     test_dataloader = load_centralized_dataset()

#     # Evaluate the global model on the test set
#     test_loss, test_acc = test(model, test_dataloader, device)

#     # Return the evaluation metrics
#     return MetricRecord({"accuracy": test_acc, "loss": test_loss})
