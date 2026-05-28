"""pytorchexample: A Flower / PyTorch app."""

# import torch
# from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
# from flwr.clientapp import ClientApp

# from pytorchexample.task import Net, load_data
# from pytorchexample.task import test as test_fn
# from pytorchexample.task import train as train_fn

# # Flower ClientApp
# app = ClientApp()


# @app.train()
# def train(msg: Message, context: Context):
#     """Train the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     batch_size = context.run_config["batch-size"]
#     trainloader, _ = load_data(partition_id, num_partitions, batch_size)

#     # Call the training function
#     train_loss = train_fn(
#         model,
#         trainloader,
#         context.run_config["local-epochs"],
#         msg.content["config"]["lr"],
#         device,
#     )

#     # Construct and return reply Message
#     model_record = ArrayRecord(model.state_dict())
#     metrics = {
#         "train_loss": train_loss,
#         "num-examples": len(trainloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"arrays": model_record, "metrics": metric_record})
#     return Message(content=content, reply_to=msg)


# @app.evaluate()
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     batch_size = context.run_config["batch-size"]
#     _, valloader = load_data(partition_id, num_partitions, batch_size)

#     # Call the evaluation function
#     eval_loss, eval_acc = test_fn(
#         model,
#         valloader,
#         device,
#     )

#     # Construct and return reply Message
#     metrics = {
#         "eval_loss": eval_loss,
#         "eval_acc": eval_acc,
#         "num-examples": len(valloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"metrics": metric_record})
#     return Message(content=content, reply_to=msg)


# pytorchexample/client_app.py

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import json

from pytorchexample.data import load_client_data
from pytorchexample.model import CustomFashionModel
import numpy as np
import io

# Create ClientApp
app = ClientApp()

def build_optimizer(model, optimizer_name: str, lr: float):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    else:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

@app.train()
def train(msg: Message, context: Context) -> Message:

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = int(context.run_config["batch-size"])
    # local_epochs = int(context.run_config["local-epochs"])
    data_dir = str(context.run_config["data-dir"])
    lr = float(msg.content["config"]["lr"])
    local_epochs = int(msg.content["config"]["local-epochs"])
    optimizer_name = str(context.run_config.get("client-optimizer", "sgd"))
    algorithm = str(context.run_config.get("client-algorithm", "fedavg"))

    malicious_ratio = float(context.run_config.get("malicious-ratio", 0.0))
    attack_type = str(context.run_config.get("attack-type", "data"))
    num_malicious = int(malicious_ratio * num_partitions)
    is_malicious  = (partition_id < num_malicious)
    
    ## load c for scaffold if needed
    if algorithm == "scaffold":
        c_global_arrays = msg.content["c_global"].to_numpy_ndarrays()
    else:
        c_global_arrays = None

    # set model
    model = CustomFashionModel()
    params = msg.content["arrays"].to_numpy_ndarrays()
    model.set_model_parameters(params)
    
    mu = float(msg.content["config"].get("mu", 0.0))
    global_params = [
        p.detach().clone()
        for p in model.parameters()
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load client's local dataset
    train_loader, _ = load_client_data(
        cid=partition_id,
        data_dir=data_dir,
        batch_size=batch_size,
    )

    # Train locally
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, optimizer_name, lr)
    print("optimizer:", optimizer_name)
    if optimizer_name == "adam" and "optimizer_state" in context.state:
        state_bytes = context.state["optimizer_state"].to_numpy_ndarrays()[0].tobytes()
        buffer = io.BytesIO(state_bytes)
        optimizer.load_state_dict(torch.load(buffer))

    if algorithm == "fedsgd":
        print("Using FedSGD")
        train_loss, train_acc = model.train_one_step(
            train_loader, criterion, optimizer, device
        )
        num_examples = batch_size   # only one batch was used
    elif algorithm == "scaffold":
        print("Using SCAFFOLD")
        # context.state persists across rounds for the same client
        if "c_local" in context.state:
            c_local_arrays = context.state["c_local"].to_numpy_ndarrays()
        else:
            # First round: initialize c_k to zeros matching model shape
            c_local_arrays = [np.zeros_like(p) for p in model.get_model_parameters()]


        w_before = model.get_model_parameters()

        train_loss, train_acc = 0.0, 0.0
        total_steps = 0
        for _ in range(local_epochs):
            train_loss, train_acc, steps = model.train_epoch_scaffold(
                train_loader, criterion, optimizer, device,
                c_local_arrays, c_global_arrays
            )
            total_steps += steps

        # Snapshot weights after training
        w_after = model.get_model_parameters()

        # Update c_local:
        # c_k ← c_k - c + (1 / η*T) * (w_before - w_after)
        # T  = total_steps * local_epochs   # total gradient steps
        T = total_steps
        lr_val = float(msg.content["config"]["lr"])

        new_c_local = [
            ck - c + (1.0 / (lr_val * T)) * (wb - wa)
            for ck, c, wb, wa in zip(
                c_local_arrays, c_global_arrays, w_before, w_after
            )
        ]

        # Save updated c_local back to persistent state
        context.state["c_local"] = ArrayRecord(new_c_local)
        num_examples = len(train_loader.dataset)
    else:
        # FedAvg
        train_loss, train_acc = 0.0, 0.0

        # for _ in range(local_epochs):
        #     if mu > 0.0:
        #         print("Using FedProx with mu =", mu)
        #         # FedProx
        #         train_loss, train_acc = model.train_epoch_fedprox(
        #             train_loader, criterion, optimizer,
        #             device, global_params, mu
        #         )
        #     else:
        #         print("Using FedAvg")
        #         # Standard FedAvg
        #         train_loss, train_acc = model.train_epoch(
        #             train_loader, criterion, optimizer, device
        #         )

        if is_malicious and attack_type == "data":
            # DATA POISONING
            print("Performing data poisoning attack (label flipping)")
            for _ in range(local_epochs):
                train_loss, train_acc = model.train_epoch_data_poison(
                    train_loader, criterion, optimizer, device
                )
        elif is_malicious and attack_type == "model":
            # MODEL POISONING
            print("Performing model poisoning attack")
            for _ in range(local_epochs):
                train_loss, train_acc = model.train_epoch(
                    train_loader, criterion, optimizer, device
                )
            poisoned_params = apply_model_poison(
                model.get_model_parameters(), attack_scale=2.0
            )
            model.set_model_parameters(poisoned_params)
        else:
            # NO POISONING
            for _ in range(local_epochs):
                if mu > 0.0:
                    print("Using FedProx with mu =", mu)
                    global_params = [p.detach().clone() for p in model.parameters()]
                    train_loss, train_acc = model.train_epoch_fedprox(
                        train_loader, criterion, optimizer, device, global_params, mu
                    )
                else:
                    print("Using FedAvg")
                    train_loss, train_acc = model.train_epoch(
                        train_loader, criterion, optimizer, device
                    )

        num_examples = len(train_loader.dataset)

    if algorithm == "scaffold":
        updated_arrays  = ArrayRecord(model.get_model_parameters())
        c_local_record  = ArrayRecord(new_c_local)
        metrics = MetricRecord({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "num-examples": float(num_examples)
        })
        content = RecordDict({
            "arrays":  updated_arrays,
            "c_local": c_local_record,
            "metrics": metrics
        })
    else:
        updated_arrays = ArrayRecord(model.get_model_parameters())
        metrics = MetricRecord({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "num-examples": len(train_loader.dataset),
            "is_malicious": float(is_malicious)
        })
        content = RecordDict({
            "arrays": updated_arrays,
            "metrics": metrics
        })

    if optimizer_name == "adam":        
        buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), buffer)
        state_array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        context.state["optimizer_state"] = ArrayRecord([state_array])

    return Message(content=content, reply_to=msg)

# model poisoning
def apply_model_poison(params: list, attack_scale: float = 5.0) -> list:
    # reverse gradient direction
    return [-p * attack_scale for p in params]

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:

    partition_id = context.node_config["partition-id"]
    batch_size = int(context.run_config["batch-size"])
    data_dir = str(context.run_config["data-dir"])

    model = CustomFashionModel()
    params = msg.content["arrays"].to_numpy_ndarrays()
    model.set_model_parameters(params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load val data
    _, val_loader = load_client_data(
        cid=partition_id,
        data_dir=data_dir,
        batch_size=batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    eval_loss, eval_acc = model.test_epoch(val_loader, criterion, device)

    metrics = MetricRecord({
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(val_loader.dataset),
    })
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)