# Regression Neural Network

A neural network project for **regression tasks** using an **evolutionary training algorithm** instead of traditional gradient-based methods.
The model evolves a population of neural networks over generations, gradually improving prediction accuracy.

# Features
- Predicts **target** using various features
- **Evolutionary algorithm** for network optimization
- Adjustable **mutation rate and mutation strength**
- **Topology mutations** (adding/removing layers and neurons)
- Supports **parallel training across CPU cores**
- Uses **NumPy matrix operations** for efficient batch predictions

# Dataset Format
Prepare a CSV dataset with named columns, eg.:
```
target, feature1, feature2, feature3, feature4, ...
```
You can also modify the **feature list** in `parameters.json`.

# Usage
### 1. Prepare the Dataset
Create a CSV file containing features and target columns.

### 2. Configure Training
This project uses **2 config files**:
- `def_parameters.json`: defines which dataset to use *(`"dataset": dataset_name`)* and provides fallback parameters
- `datasets/{dataset_name}/parameters.json`: dataset-specific parameters (these replace default ones).

For more info about configuration see *Configuration* section.

### 3. Train the Model
```
python train.py
```
During training, the evolutionary algorithm evolves a population of networks over multiple generations, improving performance over time.

### 4. Evaluate Performance
After training, the program reports the **Mean Absolute Error (MAE)**:
```
Mean Absolute Error: 18,432
```

### 5. Predict Prices
```
python frontend.py
```
Enter feature values to receive a target prediction.

# How It Works
1. A **population of neural networks** is initialized.
2. Each network predicts target prices.
3. Networks are evaluated using **Mean Absolute Error (MAE)**.
4. The best-performing networks **survive and reproduce**.
5. Offspring networks are created through **mutation**:
   - Weight mutations
   - Neuron additions/removals
   - Layer additions/removals
6. The process repeats for multiple **generations**, improving performance over time.

# Configuration (`parameters.json`)
This section explains each parameter and its usage in config files. Note that `def_parameters.json` must contain additional parameter `dataset`, which specifies dataset to be used. `def_parameters.json` does not need to include other parameters, provided they're specified in `parameters.json`.
### Network Structure
| Parameter | Description |
|----------|-------------|
| `network_size` | Initial network layer sizes |
| `max_layers` | Maximum number of layers |
| `min_layers` | Minimum number of layers |
| `max_layer_size` | Maximum neurons per layer |
| `min_layer_size` | Minimum neurons per layer |
| `max_neurons` | Maximum neurons allowed in a network |

### Evolution Settings
| Parameter | Description |
|----------|-------------|
| `population_size` | Number of networks per generation |
| `survivors_count` | Number of networks selected as parents |
| `elites_count` | Networks copied unchanged to next generation |
| `max_generations` | Maximum number of generations |
| `patience` | Stop training after this many generations without improvement |

### Mutation Settings
| Parameter | Description |
|----------|-------------|
| `mutation_rate` | Probability of mutating weights |
| `mutation_strength` | Magnitude of weight mutations |
| `mutation_strength_decay` | Mutation strength decay over generations |
| `new_layer_rate` | Probability of adding a new layer |
| `delete_layer_rate` | Probability of removing a layer |
| `new_neuron_rate` | Probability of adding a neuron |
| `delete_neuron_rate` | Probability of removing a neuron |
| `topology_mutation_treshold` | Generation after which topology mutations stop |

### Training Settings
| Parameter | Description |
|----------|-------------|
| `batch_size` | Number of random samples used per generation |
| `data_split_index` | Percentage of dataset used for training |
| `cpu_cores` | Maximum CPU cores used during training |

### Evaluation Settings
| Parameter | Description |
|----------|-------------|
| `target` | Column to predict |
| `features` | List of input features |
| `sort_key` | 1 = log-scaled MAE, 2 = raw MAE |

# Output Metrics
The model reports **Mean Absolute Error (MAE)**:
```
MAE = average(|predicted_target - actual_target|)
```
Lower values indicate better predictions.

# Project Structure
```
project-root/
│
├── datasets/                  # Available datasets
│   ├── example1/
│   │   ├── dataset.csv        # Data file
│   │   └── parameters.json    # Dataset-specific config
│   ├── example2/
│   │   ├── dataset.csv
│   │   └── parameters.json
│
├── models/                    # Trained models
│   ├── example1/
│   │   ├── genes.json         # Network weights/topology
│   │   ├── metrics.json       # Training results
│   │   └── parameters.json    # Config used for training
│   └── example2/
│       ├── genes.json
│       ├── metrics.json
│       └── parameters.json
│
├── modules/                   # Core neural network implementation
│   ├── layer.py
│   ├── network.py
│   └── normalizer.py
│
├── utils/                     # Helper utilities
│   ├── config.py
│   └── utils.py
│
├── def_parameters.json        # Global config (selects dataset)
├── train.py                   # Training script
├── frontend.py                # Prediction interface
└── README.md
```

# Notes
- Feature values are **normalized automatically**
- Target prices are **log-scaled**
- Evolutionary training may require **many generations for convergence**
- Mutation strength can **decay over time** for smoother training
- Hidden layers use **Leaky ReLU activation**
- Output layer is **linear**
- Training uses **random mini-batches**
