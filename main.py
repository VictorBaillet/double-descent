import sys
import json
import numpy as np
import os
from utils.models import TwoLayersNN, train, evaluate
from utils.ntk import *
from utils.utils import *

def main(config_file):
    # Load the configuration file
    with open(config_file, 'r') as file:
        config = json.load(file)

    # Extract configuration settings
    num_experiments = config['General']['Number of experiments']
    x_width = config['Model']['Network width']
    epochs = config['Model']['Number of epochs']
    lr = config['Model']['Learning rate']
    
    device = setup_training_device()
    train_loader, test_loader = load_dataset(config)
    width_to_results, data_path = setup_experiment_path(config)
    input_size = config["Dataset"]['Input size']['x'] * config["Dataset"]['Input size']['y']
            
    for j in range(num_experiments):
        for i, width in enumerate(x_width):
            print(width)
            net = TwoLayersNN(width=width, input_size=input_size).cuda()
            test_loss, train_loss, x_alpha = train(net, train_loader, test_loader, epochs, lr)
                
            results = width_to_results[str(width)]
            results["loss test"].append(test_loss)
            results["loss train"].append(train_loss)
            results["complexity"].append(compute_complexity(net, train_loader, test_loader, device=device))
            if len(x_alpha) > 0:
                results["alpha"].append(np.mean(x_alpha))
            results["number of parameters"] = sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad)
            width_to_results[str(width)] = results
            with open(data_path, "w") as fp:
                json.dump(width_to_results , fp) 

def setup_experiment_path(config):
    experiment_path = r'results/' +  config['General']['Name']
    data_path = experiment_path + '/' + config['General']['Sub Name'] + ".json"
    x_widths = config['Model']['Network width']
    
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    
    try:
        with open(data_path, 'r') as file:              ### It does not work ??
            width_to_results = json.load(file)
    except:
        width_to_results = {}

    for width in x_widths:
        if str(width) not in width_to_results.keys():
            width_to_results[str(width)] = {"loss test" : [],
                                       "loss train" : [],
                                       "complexity" : [],
                                       "alpha" : [],
                                       "number of parameters" : 0}

    return width_to_results, data_path

def load_dataset(config):
    if config["Dataset"]["Name"] == "CIFAR-10 cat vs dog":
        train_loader, test_loader = preprocess_CIFAR10(n_train=config["Dataset"]["Train set size"],
                                                       n_test=config["Dataset"]["Test set size"],
                                                       noise_level=config["Dataset"]["Noise level"], 
                                                       downsample_size=(config["Dataset"]['Input size']['x'], 
                                                                        config["Dataset"]['Input size']['y']))
    if config["Dataset"]["Name"] == "MNIST 0 vs 8":
        train_loader, test_loader = preprocess_MNIST(n_train=config["Dataset"]["Train set size"],
                                                     n_test=config["Dataset"]["Test set size"],
                                                     batch_size=config["Dataset"]["Batch size"],
                                                     noise_level=config["Dataset"]["Noise level"], 
                                                     downsample_size=(config["Dataset"]['Input size']['x'], 
                                                                    config["Dataset"]['Input size']['y']))

    return train_loader, test_loader

def setup_training_device():
    cuda_available = torch.cuda.is_available()

    print(f"CUDA disponible : {cuda_available}")
    
    # Affiche également le nombre de GPU disponibles et leur nom si CUDA est disponible
    if cuda_available:
        print(f"Nombre de GPU disponibles : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Aucun GPU n'est disponible. Entraînement sur CPU.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    return device

if __name__ == '__main__':
    config_file = sys.argv[1]  # Get the configuration file path from command line argument
    main(config_file)