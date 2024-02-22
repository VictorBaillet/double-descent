# Kernel-Based Complexity Measure for the Study of Double Descent in Neural Networks

We investigate double descent phenomena in neural networks, using a novel kernel-based complexity measure. Building upon the work of Curth et al. 2023.

## Installation 


1. Clone the repository:
   ```bash
   git clone https://github.com/VictorBaillet/double-descent
   cd double-descent
   ```

2. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Launch the Jupyter notebook for a hands-on experience with the results.

## Run experiments

To run the experiments on your machine :

    ```bash
    python main.py config/[experiment_name].json
    ```

## Datasets Employed

The study employs two datasets :

- **MNIST**
- **CIFAR-10**
