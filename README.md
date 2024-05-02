# Dyna-Q + with prioritized sweeping:
## Introduction:
The challenge in Reinforcement Learning lies in achieving optimal control within partially known Markov decision processes. Put simply, it involves the dynamic interplay between an agent and its environment through actions.
Each action alters the agent's state based on the environment's dynamics, yielding a numerical signal known as a reward.
The primary objective for the agent is to devise a policy — a mapping situations to actions — that maximizes the cumulative reward throughout the task.
In this context, a model of the system is represented by a probability distribution, indicating the likelihood of transitioning from state S to state S' and receiving reward R upon executing action A in state S.
While a state-action value ${Q_(\pi)}(s,a)$ represent the expected cumulative reward obtained by the agent starting in state S, following as first action A, and a gven policy thereafter.

In this domain, the Dyna-Q algorithm tackles the simultaneous learning of a policy and a model of the environment, operating online in a model-free manner, assuming the environment to be deterministic and finite.
The algorithm functions by iteratively estimating the state-action values Q(s,a) through alternating interactions with the environment and simulations.
Initially, the agent engages with the environment, gathering reward signals and new states at each interaction. These updates are utilized to refine both the state-action values and the environment model.
Subsequently, a series of simulations are conducted based on the updated environment model. These simulations further refine the state-action values, leveraging the agent's experiential knowledge of the environment.
![Dyna-Q algorithm](/images/dyna-q.jpeg "Dyna-Q algorithm")

## Environment characteristics:
In this project the proposed environment is constituted by a 2D grid world.

![Environment](/images/graph.png "Environment")

In it a series of elements can be identified:
- Obstacles: showed in gray in the picture, they limit the agent's possible motions along the environment.
- Portals: showed by circled letters in the picture, they teleport the agent from one entry to the other.
- Holes: showed in black in the picture, if encountered they end the episode, retruning a negative reward.
- Treasures: in yellow in the picture, they can be collected along the path providing a positive reward.
- starting state: shown in green in the picture.
- goal state: shown in red in the pictur, if encountered it finsh th episode, returning a positive reward.

It can be seen how both the state and action space are finite, respecting the Dyna-Q requirements.

In this setting, the agent can navigate in the four cardinal directions (up, down, left, right) in a stochastic manner. This implies that the environment introduces randomness into the agent's movements, enabling it to execute the intended action successfully in 50% of instances. In the remaining cases, with equal probability, the agent ends up in either the left or right state along the desired direction, despite its intended action.

![Stochasticity](/images/stoc_env.jpg "Stochasticity")

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

In that way at the next $\varepsilon$-greedy action selection the probability of chosing that action will increase, forcing the agent to try long untried actions in search of a variation of the environment.
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
