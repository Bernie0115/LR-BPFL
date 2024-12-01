import os
import torch
import random
import torchvision
import numpy as np
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from utils.gen_noisy_data import corrupt_images


random.seed(52)
np.random.seed(52)
num_clients = 10
num_classes = 10
class_per_client = 5
dir_path = "Cifar10-test-dir-1.0/"

# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, is_Noisy):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    noisy_train_path = dir_path + "noisy-train/"
    noisy_test_path = dir_path + "noisy-test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    # dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    # dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    x, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance,
                                    partition, class_per_client=class_per_client)

    train_data, test_data = split_data(x, y)

    if is_Noisy:
        corrupted_train_data, corrupted_test_data = corrupt_images(train_data, test_data)
        check(config_path, noisy_train_path, noisy_test_path, num_clients, num_classes, niid, balance, partition)
        save_file(config_path, noisy_train_path, noisy_test_path, corrupted_train_data, corrupted_test_data, num_clients, num_classes, statistic, niid,
              balance, partition)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid,
              balance, partition)


if __name__ == "__main__":
    niid = True
    balance = False
    partition = 'dir'
    is_Noisy = False

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, is_Noisy)
