# RL_Legged-Wheeled-sim

## Introduction
This project is a simulation of a two-legged-wheeled robot based on a reinforcement learning approach. The simulator used is **MuJoco**. The specific approach is to firstly encapsulate the robot as a standard reinforcement learning Python interface according to the **gym** style, including state feedback, action, reward function, reset, etc. The robot is then used to interact with the mujoco model through the Pytorch framework. Secondly, the **SAC** algorithm is built through pytorch framework to interact with the MuJoco model to make the robot reach the goal point.

## Environment Configuration
The project was done on Ubuntu20.04 with the following Python versions and library versions:
* Python 3.7.10
* MuJoco 210
* gym 0.21.0
* mujoco-py 2.1.2.14
* pytorch 1.10.0
* numpy 1.21.6
