"""
This file contains functions for computing Neural Tangent Kernel feature vectors and matrices, and calculating the complexity of a neural network model using these features.

Author: Victor Baillet
Repository: https://github.com/VictorBaillet/double-descent
"""

import torch
import numpy as np
import gc

def compute_feature_vector(net, y, device):
    """
    Compute the Neural Tangent Kernel feature vector for a given input.
    """
    y = y.to(device).requires_grad_(True)
    output_y = net(y)
    grad_y = torch.autograd.grad(outputs=output_y, 
                                 inputs=net.parameters(), 
                                 grad_outputs=torch.ones_like(output_y),
                                 retain_graph=True, 
                                 create_graph=True)
    grad_y_vector = torch.cat([g.contiguous().view(-1) for g in grad_y])
    return grad_y_vector

def compute_feature_matrix(net, train_loader, device):
    """
    Compute the Neural Tangent Kernel feature matrix for a training dataset.
    """
    net.eval() 
    net.to(device)

    gradients = []
    for batch in train_loader:
        x, _ = batch
        for single_x in x:  # Iterate over each example in the batch
            single_x = single_x.unsqueeze(0).to(device).requires_grad_(True)
            
            output_x = net(single_x)
            grad_x = torch.autograd.grad(outputs=output_x, 
                                         inputs=net.parameters(), 
                                         grad_outputs=torch.ones_like(output_x),
                                         retain_graph=True, 
                                         create_graph=True)
            # Flatten and concatenate gradients for each parameter
            grad_x_vector = torch.cat([g.contiguous().view(-1) for g in grad_x])
            gradients.append(grad_x_vector)

    # Stack all gradients to form the feature matrices
    phi = torch.stack(gradients)
    return phi

def compute_matrix_pseudo_inverse(phi):
    """
    Compute the pseudo-inverse of the input matrix.
    """
    phi_phi_t_pinv = torch.pinverse(phi)
    del phi
    gc.collect()
    return phi_phi_t_pinv

def compute_complexity(net, train_loader, test_loader, device):
    """
    Compute the complexity/number of effective parameters of a neural network model.

    Parameters:
    net (torch.nn.Module): The neural network model.
    train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    test_loader (torch.utils.data.DataLoader): DataLoader for test data.
    device (torch.device): The device to perform computation on.

    Returns:
    list: Median and mean of the complexity measure.
    """
    phi = compute_feature_matrix(net, train_loader, device)
    inverse = compute_matrix_pseudo_inverse(phi)
    phi = None
    torch.cuda.empty_cache()
    gc.collect()
    
    complexity = []
    for data in test_loader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda().float()
        for input in inputs:
            phi_y = compute_feature_vector(net, input, device)
            s = np.array(torch.mm(phi_y.unsqueeze(-1).T, inverse).cpu().detach().numpy())
            complexity.append(np.sum(s**2))

    phi_y, s, inverse = None, None, None
    torch.cuda.empty_cache()
    gc.collect()
    return [float(np.median(complexity)), float(np.mean(complexity))]