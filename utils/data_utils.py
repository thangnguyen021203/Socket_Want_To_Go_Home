import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

def choose_dataset(dataset_name):
    """
    Chọn dataset từ tên.

    Args:
        dataset_name: Tên dataset.

    Returns:
        Dataset: Đối tượng dataset.
    """
    if dataset_name == "MNIST":
        from torchvision.datasets import MNIST
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    elif dataset_name == "CIFAR10":
        from torchvision.datasets import CIFAR10
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    else:
        raise ValueError("Dataset not supported.")

def iid_split(dataset, num_clients):
    """
    Phân chia dữ liệu IID cho các client.
    Dữ liệu được chia ngẫu nhiên, mỗi client nhận một tập dữ liệu có kích thước bằng nhau.

    Args:
        dataset: Dataset PyTorch.
        num_clients: Số lượng client.

    Returns:
        list: Danh sách các Subset cho từng client.
    """
    num_samples = len(dataset)
    np.random.seed(2025)  # Đặt seed để đảm bảo tái lập kết quả
    all_indices = np.random.permutation(num_samples)  # Trộn ngẫu nhiên các index
    split_size = num_samples // num_clients          # Số mẫu mỗi client
    subsets = [
        Subset(dataset, all_indices[i * split_size: (i + 1) * split_size])
        for i in range(num_clients)
    ]
    return subsets


def non_iid_split(dataset, num_clients, shard_per_client=3):
    """
    Phân chia dữ liệu Non-IID cho các client.
    Mỗi client chỉ nhận dữ liệu từ một số nhãn cụ thể.

    Args:
        dataset: Dataset PyTorch.
        num_clients: Số lượng client.
        shard_per_client: Số nhãn mỗi client được gán.

    Returns:
        list: Danh sách các Subset cho từng client.
    """
    np.random.seed(2025)  # Đặt seed để đảm bảo tái lập kết quả
    num_samples = len(dataset)
    imgs_per_shard = num_samples // (num_clients * shard_per_client)

    # Tạo dictionary chứa danh sách index theo nhãn
    idxs_dict = {}
    for idx, label in enumerate(dataset.targets):
        label = label.item() if isinstance(label, torch.Tensor) else label
        if label not in idxs_dict:
            idxs_dict[label] = []
        idxs_dict[label].append(idx)

    num_classes = len(idxs_dict.keys())
    dict_users = {i: np.array([], dtype="int64") for i in range(num_clients)}

    # Gán shard ngẫu nhiên cho mỗi client
    for client_id in range(num_clients):
        rand_labels = np.random.choice(range(num_classes), shard_per_client, replace=False)
        rand_indices = []
        for label in rand_labels:
            selected_indices = np.random.choice(idxs_dict[label], imgs_per_shard, replace=False)
            rand_indices.extend(selected_indices)
        dict_users[client_id] = np.array(rand_indices)

    # Trả về danh sách các Subset
    subsets = [Subset(dataset, dict_users[i]) for i in range(num_clients)]
    return subsets


def get_dataloaders(dataset, num_clients, batch_size, split_type=1, shard_per_client=3): #split_type=1: iid, 0: non-iid
    """
    Tạo DataLoader cho từng client dựa trên cách chia dữ liệu (IID hoặc Non-IID).

    Args:
        dataset: Dataset PyTorch.
        num_clients: Số lượng client.
        batch_size: Batch size của DataLoader.
        split_type: Loại phân chia dữ liệu ("iid" hoặc "non-iid").
        shard_per_client: Số nhãn mỗi client nhận được (chỉ dùng cho Non-IID).

    Returns:
        list: Danh sách các DataLoader cho từng client.
    """
    if split_type == 1:
        subsets = iid_split(dataset, num_clients)
    elif split_type == 0:
        subsets = non_iid_split(dataset, num_clients, shard_per_client=shard_per_client)
    else:
        raise ValueError("split_type must be 'iid' or 'non-iid'.")

    dataloaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets
    ]
    return dataloaders


def get_test_dataloader(dataset, batch_size):
    """
    Tạo DataLoader cho tập test.

    Args:
        dataset: Dataset PyTorch.
        batch_size: Batch size của DataLoader.

    Returns:
        DataLoader: DataLoader cho tập test.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)