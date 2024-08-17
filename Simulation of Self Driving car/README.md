# Self-Driving Car Simulation 🚗🤖

Welcome to the **Self-Driving Car Simulation** project! This project leverages the power of **Reinforcement Learning (RL)** and the **Kivy framework** to simulate a self-driving car that learns how to navigate through an environment. By training an agent, the car learns to drive autonomously through the provided scenarios.

## 🎯 Project Overview

This project simulates a self-driving car that makes decisions using a reinforcement learning algorithm. The simulation is rendered using the **Kivy** framework, which provides an intuitive and flexible environment for designing interactive applications.

Through trial and error, the car improves its performance by maximizing its reward through a learning process, following the principles of RL.

## 🚀 Features

- **Reinforcement Learning Algorithm**: Implements Q-learning or Deep Q-learning to teach the car how to drive autonomously.
- **Dynamic Environment**: The simulation includes a customizable environment where the car learns to navigate.
- **Visualization**: Real-time visualization using Kivy to render the car and its surroundings.
- **Modular Codebase**: Easily extensible and modular codebase for adding new features or improving the model.



## 🧠 Reinforcement Learning

The self-driving car is trained using reinforcement learning, where it tries to maximize cumulative rewards. The algorithm enables the car to explore and exploit the environment while improving its driving capabilities over time.

Key RL concepts used:
- **State**: The car's position, orientation, and sensor inputs.
- **Actions**: The car's possible movements (e.g., accelerate, turn left, turn right).
- **Reward**: Positive for progressing toward the goal and negative for collisions or other undesirable behaviors.

## 🛠️ Installation

To get started with the project, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/laasriy/Data-Science.git
    cd 'Simulation of Self Driving car/'
    ```

2. **Set up the virtual environment** (optional but recommended):

    ```bash
    conda env create -f env.yml     # to create the environement to use

    conda activate self-driving-ca      # Activate the environement and installation of dependencies
    
    ```

4. **Run the simulation**:

    ```bash
    python map.py
    ```

## 📋 Usage

1. Once the simulation starts, the self-driving car will begin to explore its environment.
2. Watch as it learns to navigate over time, avoiding obstacles and reaching its goals.
3. Draw the lines in the interface to use as coordinates in Q-learning step.
4. Adjust parameters in the configuration files to tweak the environment or the learning algorithm.

## 🛠️ Technologies Used

- **Python**: Core language for the logic and reinforcement learning algorithms.
- **Kivy**: For GUI and simulation rendering.
- **TensorFlow/PyTorch** *(optional)*: If using deep learning for the RL agent.
- **NumPy**: For numerical operations.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model, add new features, or fix bugs, feel free to submit a pull request.

1. Fork the project
2. Create your feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Submit a pull request

Please ensure your code adheres to the project's coding standards.



---

Happy Coding! 🚗✨
