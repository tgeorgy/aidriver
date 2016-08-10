# AiDriver

Although the heading is very promising, the code simply implements Deep Q Learning solution for Mail.ru AI contest.

## Dependencies
Python 2.7
- numpy (tested on 1.11.1)
- OpenCV (tested on 2.4.8)
- Requests (tested on 2.2.1)

Lua+Torch
- nn
- nngraph
- cunn
- cudnn
- waffle (https://github.com/benglard/waffle)

## Environment
Environment engine is written in Python 2.7 by Mail.ru Company. Original code can be found here (https://github.com/Russian-AI-Cup-2015).

Unique code is mostly in this file *environment/python2-cgdk/MyStrategy.py*. This is where I define rewards, build state image, call agent and pass action to the game engine.

## Agent
Agent is written using lua+torch. It works as a web service - gets requests from environment with state information and responds with action index. 

Although formally the model is DDQN with prioritized sampling, there are also some changes.

Neural network consists of 3 layers convolutional (parallel 3x3 and (9x1 -> 1x9) filters followed by BatchNormalization and ReLU) and two fully connected layers.

## Running
To run training first you need to start agent web service
```
th agent/train 
```
After that simply run bash script that will iteratively generate game parameters, run the game and user strategy script.
```
./run-all.sh
```
Game logs will be saved here *environment/local-runner/logs/* 

## Results
This video shows how agent behavior changes during training (captured same map).

[![AIDRIVER](https://img.youtube.com/vi/Xd6MY3QkS9o/0.jpg)](https://www.youtube.com/watch?v=Xd6MY3QkS9o)

Overall training took roughly 3 days on Nvidia GTX 970