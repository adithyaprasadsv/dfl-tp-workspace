from pathlib import Path
import tomli
from pytorchexample.data import generate_distributed_datasets

def main():
    cfg = tomli.loads(
        Path("pyproject.toml").read_text()
    )["tool"]["flwr"]["app"]["config"]

    num_clients = int(cfg["num-clients"])
    alpha = float(cfg["alpha-dirichlet"])
    data_dir = str(cfg["data-dir"])

    print(f"Generating data for {num_clients} clients, alpha={alpha} to {data_dir}")
    generate_distributed_datasets(k=num_clients, alpha=alpha, save_dir=data_dir)
    print("Done.")

if __name__ == "__main__":
    main()