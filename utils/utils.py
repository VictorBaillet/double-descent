import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pickle

import random


def add_noise_to_labels(y, noise_level=0.1, num_classes=10):
    """
    Adds noise to the labels in the dataset.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset to add noise to.
    noise_level (float): The fraction of labels to add noise to (between 0 and 1).
    num_classes (int): The number of classes in the dataset.

    Returns:
    torch.utils.data.Dataset: The dataset with noisy labels.
    """
    noisy_y = []
    for label in y:
        if random.random() < noise_level:
            # Randomly change the label to a different class
            new_label = random.randint(0, num_classes)
            # Ensure new label is different from the original
            while new_label == label:
                new_label = random.randint(0, num_classes)
            noisy_y.append(new_label)
        else:
            noisy_y.append(label)
    
    return noisy_y

def preprocess_MNIST(n_train=4000, noise_level=0.2, zero_vs_all=False, downsample_size=None):
    transform = None
    if downsample_size is not None:
        transform = transforms.Compose([
            transforms.Resize(downsample_size),
            transforms.ToTensor()
        ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    X_train = []
    y_train = []
    for i in range(len(trainset)):
        image, label = trainset[i]
        X_train.append(np.array(image))
        y_train.append(float(label==0) if zero_vs_all else float(label))

    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=n_train, stratify=y_train, random_state=42)

    num_classes = 2 if zero_vs_all else 10
    noisy_y_train = add_noise_to_labels(y_train, noise_level=noise_level, num_classes=num_classes)

    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(noisy_y_train, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True) 

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    X_test = []
    y_test = []
    for i in range(len(testset)):
        image, label = testset[i]
        X_test.append(np.array(image))
        y_test.append(float(label==0) if zero_vs_all else float(label))

    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True) 

    return train_loader, test_loader

def preprocess_CIFAR10(n_train=1000, n_test=1000, noise_level=0, downsample_size=(8, 8)):
    # Define transformations
    transform_list = [transforms.Grayscale(num_output_channels=1)]  # Convert images to grayscale
    if downsample_size is not None:
        transform_list.append(transforms.Resize(downsample_size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Function to filter cat (class 3) and dog (class 5) images
    def filter_cats_dogs(dataset):
        X = []
        y = []
        for image, label in dataset:
            if label == 3 or label == 5:  # Cat or Dog
                X.append(np.array(image))
                y.append(float(label == 3))  # Cat=1, Dog=0
        return X, y

    # Filter trainset and testset
    X_train, y_train = filter_cats_dogs(trainset)
    X_test, y_test = filter_cats_dogs(testset)

    # Downsample training and testing data
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=n_train, stratify=y_train, random_state=42)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=n_test, stratify=y_test, random_state=42)

    # Add noise to labels (if needed, adjust or remove this part)
    noisy_y_train = add_noise_to_labels(y_train, noise_level=noise_level, num_classes=2)

    # Convert to tensors
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(noisy_y_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    return train_loader, test_loader

def load(filename):
    with open("data/" + filename, "rb") as fp:   
        file = pickle.load(fp)
    return file

def save(file, filename):
    with open("data/" + filename, "wb") as fp:   #Pickling
        pickle.dump(file, fp)