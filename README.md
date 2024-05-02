# Dyna-Q + with prioritized sweeping:
## Introduction:
The problem of Reinforcement Learning is the optimal control of incompletely-known Markow decision processes, in other words it consider the interaction of an agent with the environment through actions.
Each action modify the state in which the agent is based on the environment model, providing a numeric signal called reward.
The main purpose of the agent is to find a policy - a mapping from situations to actions - to obtain the maximum cumulative reward over the task.
Under this scenario a model of the system is represented by a probability distribution expressing the probability for the action to end in the state S' and collect the reward R once perfored the action A in state S.
Instead as state-action value Q(s,a) we mean 

In this field the Dyna-Q algorithm adresses the learning of a policy together with a model of the environment, assumed to be deterministic and finite, online in a modelfree fashion.
The algorithm works by the estimation of the state-action values Q(s,a), alernating interactions with the environment, with simulations.
In the first part the environment is experienced by the agent, collecting at each interaction the reward signal and new state, used to update the state-action values together with the environment model.
After that a series of simulations is perfomed, based on the updated environment, continuing to improve the state-action values based on the agent experience of the environment. 

![Dyna-Q algorithm](/images/dyna-q.jpeg "Dyna-Q algorithm")

## Environment characteristics:
In this project the proposed environment is constituted by a 2D grid world.

Immage environment

In it a series of elements can be identified:
- Obstacles: showed in gray in the picture, they limit the agent's possible motions along the environment.
- Portals: showed by circled letters in the picture, they teleport the agent from one entry to the other.
- Holes: showed in black in the picture, if encountered they end the episode, retruning a negative reward.
- Treasures: in yellow in the picture, they can be collected along the path providing a positive reward.

It can be seen how both the state and action space are finite, respecting the Dyna-Q requirements.

In this environment the agent is allowed to move in the four cardinal directions (up, down, left, right) in a stochastic way.
In detail the environment perturbate the agent motion, allowing it to realize the desired action in 50% of the cases, while making it land on the left or right state respect the desired action in the rest of the cases with same probability.

Immagine stochasticity of the environment

As we can see the stochasticity of the environment breaks the Dyna-Q requirement of a deterministic environment, moreover the disappearance of the treasure once collected shows the dynamic feature of the system, increasing the environment complexity.

## Stochasticity handling:
In order to deal with the stochasticity of the environment the Dyna-Q algorithm requires some modification.
In particular in the model learning it will be necessary to save for each state/action couple the new state/reward combination followed by their occurence.
In this way in the successive simulation phase the algorithm will be able to sample from such probability distirbution, reproducing loyally the environment behavior.

## Dynamicity handling:
To cope with the environment dynamicity, imposed by the disappearence of the treasures, it is necessary to improve the algorithm saving during the interactions with the environment the time step at which the given state/action couple was tried for the last time.
This data will then be used in the simulation phase where the simulated reqard will be increased by a bonus proportional to the time passed from the last time that state/action couple was tried, following the formula:

$R_{+} = R + c\sqrt{\tau}$

where:
- R: the reward experienced in the realworld interaction.
- c: a costant tuning weighting the bonus.
- $\tau$ : the time elapsed form the last occurence of the given state/action couple.
The general idea is that over the simulation phase the increased simulated reward will inflate the state-action value of the couples untried for a long time.
In that way at the next \varepsilon -greedy action selection the probability of chosing that action will increase, forcing the agent to try long untried actions in search of a variation of the environment.
This modification of the algorithm leads to the Dyna-Q + algorithm.
# Dyna-Q + with prioritized sweeping:
introduction on what is the dyna Q
brief explanation of what we now of the environment
need for modifications due to the stochasticity of the env and its dynamic
dyna Q+
make it more efficient prioritized sweeping
code analysis:
model data structure
functions
parameters choice
