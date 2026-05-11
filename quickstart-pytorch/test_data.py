# test_data.py  (temporary, delete after)
from pytorchexample.data import generate_distributed_datasets, load_client_data

generate_distributed_datasets(k=3, alpha=0.5, save_dir="./data_test")

train_loader, val_loader = load_client_data(cid=0, data_dir="./data_test", batch_size=32)
batch = next(iter(train_loader))
print("Image batch shape:", batch[0].shape)   # expect: [32, 1, 28, 28]
print("Label batch shape:", batch[1].shape)   # expect: [32]
print("Train batches:", len(train_loader))
print("Val batches:",   len(val_loader))