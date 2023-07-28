# Reinforcement learning
This repo contains various work done during the CAS Advanced Machine Learning in 2022. 
The goal was to create a snake game. For this endeavour, various other implementations and gyms have been tested.

## Game Snake
The Snake Game is a classic arcade game that remains popular for its simplicity and addictiveness. 
In this game, you take control of a snake that moves around the screen, aiming to eat food and grow longer. 
The objective is to keep the snake alive for as long as possible without colliding with the walls or itself.

## How to Play

Run 
```python
python play_Snake_Human.py
```

in your console. 

Controls: Use the keys (D,S,W,A) to control the snake's movement. 
Be mindful of not making rapid 180-degree turns, as this can lead to collisions with the snake's body.

Eating Food: As the snake moves around, it will encounter food represented by small dots. Eating food increases the snake's length by one unit and earns you points.

Growing Longer: Each time the snake eats food, its tail grows longer. Managing a long snake becomes more challenging as it requires precise navigation to avoid collisions.

Game Over: The game ends if the snake hits the screen boundaries or collides with itself. At that point, your final score is displayed, and you have the option to restart the game and beat your previous record.

Scoring
Eating Food: Every time the snake consumes food, you gain points. The score depends on the specific implementation or rules of the game.

Increasing Difficulty: As the game progresses and the snake grows longer, the difficulty level increases. The snake's speed may escalate, making it more challenging to control and avoid collisions.

# Project Contents
This repository contains the following:

- Implementations of the Snake Game using using own [gym](https://www.gymlibrary.dev/index.html).
  - [Vanilla Gym of Snake](https://github.com/schlafel/ReinforcementLearning/tree/master/snake_gym)
  - [Simplified Snake Version](https://github.com/schlafel/ReinforcementLearning/tree/master/snake_gymDirected) (Direct returning is not allowed)
- Exploration of various environments using OpenAI Gym for training and evaluation purposes. 
  - [Actor2Critic Implementation for OpenAI-Gym LunarLander](https://github.com/schlafel/ReinforcementLearning/tree/master/a2c_lunarLander)
  - [Actor2Critic Implementation for OpenAI-Gym CartPole](https://github.com/schlafel/ReinforcementLearning/tree/master/a2c_model_cartpole)
- Code and resources related to the CAS Advanced Machine Learning course.


# Reinforcement implementation details
During the project, various different RL-Setups have been tested.  
## Vanilla Agent 
The action space for the "vanilla-snake" consists of 3 different actions:

*Move up
*Move down
*Move right


The action space has 11 parameters:
1. Danger ahead 
2. Danger to the right
3. Danger to the left
4. Move direction left
5. Move direction right
6. Move direction up
7. Move direction down
8. Food location to the left
9. Food location to the right
10. Food location up
11. Food location below





![snake_vanilla](https://user-images.githubusercontent.com/68279686/209550638-11ad3b49-c370-42f7-943f-52fe30a1f719.png)


## Snake-V2 ##
This implementation has a grid that is passed as an observation space. The default argument is a 20x20 size grid. 




  
# Getting Started
To get started with the Snake Game and reinforcement learning implementations, please refer to the documentation and code available in this repository. 
You can find specific instructions on setting up the environment, running the game, and experimenting with reinforcement learning algorithms.




# Results #

Modelling the grid-snake was done using a convolutional neural network (DQN). The model had the following parameters:
![image](https://user-images.githubusercontent.com/68279686/209550311-27a923b1-939c-4f0d-997b-c3975c20dda0.png)

A result after training after ~60'000 games is shown in the video below:

[![Watch the video](video_thumbnail.jpg)](Media/209550460-1d87442f-1d5c-47ba-a667-b0ed2e393c32.mp4)

https://user-images.githubusercontent.com/68279686/209550460-1d87442f-1d5c-47ba-a667-b0ed2e393c32.mp4


<video width="640" height="360" controls>
    <source src="Media/209550460-1d87442f-1d5c-47ba-a667-b0ed2e393c32.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

# Conclusion
The Snake Game project presents a practical and entertaining application of reinforcement learning in a classic arcade setting. 
By exploring various techniques and environments, this work aims to contribute to the understanding and advancement of reinforcement learning methods.

Feel free to explore the repository and have fun playing the Snake Game! 🐍🍎🎮
Within this project, a new gym implementation for the game Snake has been created. 
The snake-gym has various different implementations