"""
This file includes functions for adding noise to labels and images, preprocessing datasets (MNIST and CIFAR10), and loading/saving experimental data.

Author: Victor Baillet
Repository: https://github.com/VictorBaillet/double-descent
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pickle
import json
import random


def random_class_noise_to_labels(y, noise_level=0.1, num_classes=10):
    """
    Adds classification noise to the labels in the dataset.
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

def continuous_noise_to_labels(y, noise_level=0.1, num_classes=10):
    """
    Adds Gaussian noise to the labels in the dataset.
    """
    noisy_y = []
    for label in y:
        # Add noise to the label
        noise = (random.random() - 0.5) * 2 * noise_level
        new_label = label + noise

        # Ensure the new label is within the valid range
        new_label = max(0, min(num_classes - 1, new_label))

        noisy_y.append(new_label)

    return noisy_y

def add_noise_to_images(images, noise_level=0.1):
    """
    Adds Gaussian noise to the images in the dataset.
    """
    noisy_images = []
    for image in images:
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        # Clipping to maintain pixel value range between 0 and 1
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_images.append(noisy_image)

    return np.array(noisy_images)

def preprocess_MNIST(n_train=4000, n_test=3000, batch_size=128, noise_level=0, downsample_size=None):
    # Define transformations
    transform = None
    if downsample_size is not None:
        transform = transforms.Compose([
            transforms.Resize(downsample_size),
            transforms.ToTensor()
        ])
    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter classes 0 and 8
    X_train = []
    y_train = []
    for i in range(len(trainset)):
        image, label = trainset[i]
        if label==0 or label==8:
            X_train.append(np.array(image))
            y_train.append(float(label==8)) # 8->1, 0->0

    X_test = []
    y_test = []
    for i in range(len(testset)):
        image, label = testset[i]
        if label==0 or label==8:
            X_test.append(np.array(image))
            y_test.append(float(label==8))

    # Downsample training and testing data
    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=n_train, stratify=y_train, random_state=42)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=n_test, stratify=y_test, random_state=42)

    # Add noise to labels 
    y_train = random_class_noise_to_labels(y_train, noise_level=noise_level, num_classes=2)
    #X_train = add_noise_to_images(X_train, noise_level=noise_level)

    # Convert to tensors and dataloaders
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 


    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

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

    # Add noise to labels 
    noisy_y_train = random_class_noise_to_labels(y_train, noise_level=noise_level, num_classes=2)

    # Convert to tensors and dataloaders
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(noisy_y_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    return train_loader, test_loader

def load(filename):
    with open("data/" + filename, "rb") as fp:   
        file = pickle.load(fp)
    return file

def save(file, filename):
    with open("data/" + filename, "wb") as fp: 
        pickle.dump(file, fp)

def unpack_results(config):
    """
    Unpacks experiment results from a JSON file.
    """
    
    # Constructing the file path from the configuration
    experiment_path = f"results/{config['General']['Name']}"
    data_path = f"{experiment_path}/{config['General']['Sub Name']}.json"

    # Reading experiment results from the JSON file
    with open(data_path, 'r') as file:
        width_to_results = json.load(file)

    # Initializing lists to store unpacked results
    x_loss_train, x_loss_test, x_complexity, x_alpha = [], [], [], []
    # Sorting the widths for consistent order
    x_width = np.sort(np.array(list(width_to_results.keys()), dtype=int))
    
    # Iterating over each width and extracting corresponding results
    for width in x_width:
        results = width_to_results[str(width)]

        # Only proceed if there are training loss results
        if results["loss train"]:
            x_loss_train.append(np.mean(results["loss train"]))
            x_loss_test.append(np.median(results["loss test"]))

            # Extract complexity and alpha values
            complexity = np.array(results.get("complexity", []))
            alpha = np.array(results.get("alpha", []))
            if complexity.size > 0:
                x_complexity.append(np.median(complexity[:, 0]))
            if alpha.size > 0:
                x_alpha.append(np.median(alpha))

    return x_loss_train, x_loss_test, x_complexity, x_width, x_alpha