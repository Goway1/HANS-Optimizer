# Hierarchical Adaptive Momentum with Nesterov Scaling (HANS): A Neural Network Optimizer

One of the most commonly used optimization algorithms for neural networks is ADAM. By adding Nesterov Scaling and modifying ADAM's adaptive momentum, HANS was created. The novel addition is the "Hierarchy" that was implemented to the adaptive momentum; instead of calculating a singular momentum based on all the past gradients, HANS calculates multiple based on different time scales. In other terms, HANS is both highly responsive and stable by using short-term and long-term momentums. This repository introduces HANS and provides methods of comparison to ADAM and SGD. 

## Features 
- HANS Optimizer
- CNN and ResNet Models
- Utilities for data processing, training, and debugging
- Configuration system `config.py` (change epochs, learning rate, batch size, model) 

## Installation
```bash
git clone https://github.com/Goway1/HANS-Optimizer.git
cd HANS-Optimizer
pip install -r requirements.txt
```
## Contributions
You are more than welcome to contribute!

## License
This project is licensed under the MIT License. Please check `LICENSE` for more details.
