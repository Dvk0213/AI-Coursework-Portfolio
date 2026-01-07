# AI Coursework Portfolio

This repository contains implementations of fundamental Artificial Intelligence and Deep Learning algorithms, developed during my coursework at the **University of Wisconsin-Madison** (CS 540: Introduction to AI). 

The projects demonstrate my proficiency in **PyTorch**, **Reinforcement Learning**, and **Adversarial Search**, focusing on building models and agents from scratch to understand their underlying mechanics.

## 📂 Project Highlights

### 1. LeNet-5 CNN Implementation (PyTorch)
* **Path:** `computer_vision/lenet_cnn_classifier.py`
* **Description:** A manual implementation of the LeNet-5 Convolutional Neural Network architecture for image classification.
* **Key Technical Features:**
    * **Architecture Design:** Constructed the network using `torch.nn.Module`, defining Convolutional layers, Max Pooling, and Fully Connected layers.
    * **Training Loop:** Implemented the full optimization pipeline from scratch, including:
        * Forward propagation
        * Loss computation (CrossEntropyLoss)
        * **Manual gradient handling** (`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`) to demonstrate understanding of Autograd.
    * **Dimensionality Management:** Explicit handling of tensor shape transformations between convolutional and linear layers.

### 2. Tabular Q-Learning Agent
* **Path:** `reinforcement_learning/rl_frozenlake_agent.py`
* **Description:** An off-policy TD control agent trained to solve the gymnasium `FrozenLake-v1` environment.
* **Key Technical Features:**
    * **Q-Table Update:** Implemented the core Bellman update equation: $Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a'))$.
    * **Exploration vs. Exploitation:** Designed an $\epsilon$-greedy strategy with epsilon decay to balance exploring the environment and exploiting learned policies.
    * **State-Action Mapping:** Managed state-action pairs using Python dictionaries for efficient lookup.

### 3. Minimax Game Agent (Teeko)
* **Path:** `search_algorithms/minimax_game_agent.py`
* **Description:** An adversarial AI agent capable of playing the strategy board game Teeko.
* **Key Technical Features:**
    * **Minimax Algorithm:** Implemented a recursive depth-limited search to evaluate game states.
    * **Heuristic Evaluation:** Designed a custom evaluation function that scores board states based on piece position, control of the center, and proximity to winning configurations (4-in-a-row).
    * **Successor Generation:** Logic to handle different game phases (Drop Phase vs. Movement Phase).

---

## 🛠 Skills & Tools
* **Languages:** Python 3
* **Deep Learning:** PyTorch, Torchvision
* **Reinforcement Learning:** Gymnasium
* **Scientific Computing:** NumPy
* **Version Control:** Git

## 🚀 Usage

To run the Convolutional Neural Network training:
```bash
cd computer_vision
pip install -r requirements.txt
python lenet_cnn_classifier.py
```

To train the Q-Learning agent:
```bash
cd reinforcement_learning
pip install -r requirements.txt
python rl_frozenlake_agent.py
