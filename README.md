# RL_Legged-Wheeled-sim

## Introduction
This project is a simulation of a two-legged-wheeled robot based on a deep reinforcement learning approach. The simulator used is **MuJoco**. The specific approach is to firstly encapsulate the robot as a standard reinforcement learning Python interface according to the **gym** style, including state feedback, action, reward function, reset, etc. The robot is then used to interact with the mujoco model through the **Pytorch** framework. Secondly, the **SAC** algorithm is built through Pytorch framework to interact with the MuJoco model to make the robot reach the goal point.

## Environment Configuration
The project was done on Ubuntu20.04 with the following Python versions and library versions:
* Python 3.7.10
* MuJoco 210
* gym 0.21.0
* mujoco-py 2.1.2.14
* Pytorch 1.10.0
* numpy 1.21.6

## Clarification
The project consists of three folders, a main file, a test file, and four other function files. These programs can be divided into three parts: the robot environment, the results, and the algorithms.

### 1. Environmental files
The robot simulation environment is mainly written in the **envs** folder, which consists of the **asset** folder, the file named `biped.py`, and the file named `register.py`.

`biped.py` encapsulates environment reset, step environment update (including reward function), status feedback, etc. The main function is to encapsulate the information of the robot inside the mujoco emulator through Python interface. `register.py` is the bootstrap file to guide the project to locate biped.py.

The **asset** folder mainly contains the mujoco simulator, `Legged_wheel.xml`, `Legged_wheel.xml1`, `Legged_wheel.xml2`, `Legged_wheel.xml3` are four different simulation scenarios. **meshes** folder stores the robot parts in **stl** form.

### 2. Results file of the training
The trained neural network model is recorded in the **models** folder and called by `test.py`. The runtime data is recorded in the **runs** folder. To view the data, go to the runs folder and use the command:
```
tensorboard --logdir=[filename]
```
![image](https://github.com/hrxsd/RL_Legged-Wheeled-sim/blob/master/legged_wheeled_mujoco/023-08-02%2016%3A23%3A54.png)

### 3. Reinforcement learning algorithm file
`main.py` is the main training program, the algorithm is **SAC (`sac.py`)** and the neural network model is written in Pytorch **(`model.py`)**. 
Since SAC is offline reinforcement learning, it requires **replay_memory (`replay_memory.py`)**. `utils.py` defines some script functions.

## Usage
1. Install MuJoco 210
2. Install Anaconda
3. Create a conda environment
4. Install mujoco-py, gym and Pytorch
5. Download the project locally and run the following command for saving the running data:
```
cd RL_Legged-Wheeled-sim/legged_wheeled_mujoco
mkdir runs
```
6. Start a training session.
```
python main.py
```
7. Testing the trained model
```
python test.py
```

