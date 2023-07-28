<h1>Reinforcement learning applied to Snake</h1>
Within this project, a new gym implementation for the game Snake has been created. 
The snake-gym has various different implementations

## Vanilla Snake ##
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



# Results #

Modelling the grid-snake was done using a convolutional neural network (DQN). The model had the following parameters:
![image](https://user-images.githubusercontent.com/68279686/209550311-27a923b1-939c-4f0d-997b-c3975c20dda0.png)

A result after training after ~60'000 games is shown in the video below:



https://user-images.githubusercontent.com/68279686/209550460-1d87442f-1d5c-47ba-a667-b0ed2e393c32.mp4

