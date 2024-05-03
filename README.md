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

## Optimization:
A further optimization of the proposed algorithm may be realized with the adoption of the prioritized sweeping approach, shown in picture:
In essence, this approach involves scrutinizing the state-action value updates during both the simulation and interaction phases. If a particular update exceeds a predefined threshold, the corresponding state-action pair is enlisted in an ordered list based on the magnitude of the update.
Indeed, due to the low reward density of the system, many time the state-action update would run whith a zero or quasi-zero reward and/or a small update given by the state-action value of the next state. To avoid unnecessary evaluations, updates are initially assessed and then processed at a later stage.
Additionally in the state-action value update we would like to prioritize the largest ones, that will meaningfully change its value, and consiquently the policy; following this reasoning the update entity cover the role of a priority index.
That list will then be processed in a second step, starting from the element with highest priority.

Finally, noteworthy updates stemming from rewards on specific state-action pairs induce significant variations in their respective state-action values. This effect can propagate to other state-action pairs predicted to lead to the newly updated state-action pair, with their priorities assessed and inclusion in the list when warranted.

![Prioritized sweeping](/images/prioritized_sweeping.jpg "Prioritized sweeping")

## Implementation:
In this section we will analyze in detail the implementation of the proposed algorithm.
After an inizialization of the algorithm parameters and environment's informations, needed both for the algorithm and for the final plotting of the results, different methods are presented, this section will start from the resentation of the main method, digging deeper in the others methods once found.
# some words on the algorithm properties and the model data structure

```
    def dyna_q(self):
        # initialization
        time_out = self.rows * self.column * 10
        t_out = self.rows * self.column
        t = 0
        old_t = 0
        cum_rew = 0
        PQueue = []
        save = False
        
        for i in range(self.n_episode):
            done = False
            cum_rew = 0
            traj = []
            if (i == (self.n_episode - 1)): save = True

            s = self.env.reset().observation # observe the initial state
            state = (s[0], s[1])
            self.start_state = state

            while not done:
                t += 1
                
                action = self.eps_greedy_action(state)
                timestep = self.env.step(action)
                n_s = timestep.observation
                n_state = (n_s[0], n_s[1])
                reward = float(timestep.reward)
                cum_rew += reward
               
                if (save == True): traj.append([state, action])
                
                self.model_update(state, action, t, reward, n_state)
                PQueue = self.pq_update(state, action, reward, n_state, PQueue)
     
                sim_count = 0
                while ((PQueue != []) and (sim_count <= self.n_simulations)):
                    sim_count += 1
                    sim_state, sim_action = PQueue[0][0:2]
                    del PQueue[0]
                    new_sim_state, sim_reward = self.simulation(sim_state, sim_action, t)
                    
                    self.q_update(sim_state, sim_action, sim_reward, new_sim_state)
                    
                    pred = self.SA_predict(sim_state)
                    for pr in pred:
                        PQueue = self.pq_update(pr[0], pr[1], pr[2], sim_state, PQueue)
                        #print("PQ F", PQueue)
                
                state = n_state

                if ((timestep.is_last()) or (t >= t_out)): # if we reached the end of the episode or the step limit (to prevent loops)
                    t_out = t + time_out
                    print("episode: ", i, " steps: ", t - old_t, " cum reward: ", cum_rew)
                    self.finish_rt .append(t - old_t)
                    old_t = t
                    self.cumm_rew.append(cum_rew)
                    done = True
        
        return traj
```
Following the pseudo code presented above first some variable are initializated, secondly the "loop forever" in the pseudocode has been substituted with a for loop over a decided number of episodes.
After that the initial state is observed, and stored in the variable state.
Based on it an $\varepsilon$-greedy action is picked thanks to the `eps_greedy_action()` method:
```
    def eps_greedy_action(self, state): # chose an epsilon greedy action for the given state
        q = []
        actions = self.env.valid_actions(state) # get all the possible actions
        for act in actions: # for all the possible actions
            try:
                q.append(self.model[state, act][2]) # if we've encountered it add it to the q list
            except:
                q.append(0) # else init it to zero

        if (np.random.uniform() < self.epsilon): # not greedy
            del actions[q.index(max(q))] # remove the greedy action
            act = np.random.choice(actions) # pick randomly among the others
        else: # greedy
            act = actions[q.index(max(q))] # pick the greedy action
        
        return act
```
This method takes in input the current state, based on which it obtain all the possible actions thanks to the environment method `valid_actions()`.
Further inside the for loop the algorithm search for the state-action values of the possible actions inside the model, if the state/action couple has not been explored yet it provide an initial value of 0.
Then accordingly to the $\varepsilon$ previously set during the initialization, the method pick an action greedy with probability 1 - $\varepsilon$, or not greedy with probability $\varepsilon$ with respect to the state-action values, that is finally returned.

Contineuing on the main method the action is taken on the environment and the new state and reward are collected.
The reward is summed to the cumulative reward variable, used later as an evaluation metric and an if condition is used for the collection of the last trajectory followed by the agent.
Subsequently the model is updated by means of the method `model_update()`:
```
    def model_update(self, state, action, time, reward, new_state):
        seen = False
        try: # if we've already encountred the state action couple
            self.model[state, action][0] += 1 # increase the occurence of the given state action couple
            self.model[state, action][1] = time # update the last time step we encountered the state action/couple            
            for i in range(len(self.model[state, action][3])): # for all the new state/reward encountered after the state action couple
                if (self.model[state, action][3][i][0] == new_state) and (self.model[state, action][3][i][1] == reward): # if we've already encountered the new state reward combination
                    self.model[state, action][3][i][2] += 1 # increase its coourence
                    seen = True 
            
            if seen == False: # if we didn't encounter this new state reward couple
                self.model[state, action][3].append([new_state, reward, 1]) # append the couple over the new states for the state action couple
        
        except: # else init it
            self.model[state, action] = [1, time, 0, [[new_state, reward, 1]]]
```
This method takes in input the current state, the action performed, the current time step, the obtained reward and new state.
These data are used inide the method for the update of the model (under the `try:` function) or its initialization (under the `except:` function).
in the first case the number of occurences of the given current state/action couple is increased, it's time field is updated with the time of its last occurence, its field grouping all the new states, reward and related occurence is first scanned to check if the new state/reward couple is already present, and in this case its occurence number is updated, else it is added to the list.
In the other case the new state/action couple is added to the model with the related information.
It's important to note how this method is where the actual inizialiation of the model takes place, adding a new model element only when encountered, optimizing the memory consumption.

Back to the main method, after the model update the queue is initialized/updated by the `pq_update()` method:
```
    def pq_update(self, state, action, reward, new_state, PQ):
        done = False
        q = []
        actions = self.env.valid_actions(new_state) # get all the possible actions
        for act in actions: # for all the possible actions
            try:
                q.append(self.model[new_state, act][2]) # if we've encountered it add it to the q list
            except:
                q.append(0) # else init it to zero
        max_q = max(q) # check the maximum one
        p = abs(reward + max_q - self.model[state, action][2]) # get the q update

        if (p > self.theta):
            for pq_e in PQ: # for all the element of the list
                if (pq_e[0] == state) and (pq_e[1] == action): # if i've already the element in the list update the priority
                    if (pq_e[2] < p):
                        pq_e[2] = p
                    done = True
                    
            if (done == False): # else put it in the list
                PQ.append([state, action, p])

            PQ.sort(reverse = True, key = lambda pq: pq[2]) # order the list
        
        return PQ
```
This method takes in input the same variable of the model update one, except for the current time step and the queue list, previously initializated to empty.
It proceed by a search of the maximum state-action value of the new state, similarly to what did in `eps_greedy_action()` for the current state.
The maximum state-action value of the next state is used together with the current state action one and the reward for the evaluation of the state/action priority `p`.
The priority is then metched with the threshold and if bigger the state/action pair is first searched to see if already presenty in the list, and in this case if the new evaluated priority is bigger it is updated. else, if the state/action is not present in the queue it's added.
Finally the list is sorted and returned.

The obtained priority list is then used in the while loop to perform the simulations: in details the while loop run for a certain number of simulations, imposed by the `n_simulations` variable, as long as the priority list is not empty.
inside it the simulation state and action, `sim_state`, `sim_action` are taken from the top of the list (where they are removd) and fed to the `simulation()` method:
```
    def simulation(self, state, action, t):
        sa_time = t - self.model[state, action][1] # get the number of timesteps from the last time we tried that state action
        pos_outcomes = [ns_r[0:2] for ns_r in self.model[state, action][3]] # get the possible new state/reward
        pos_probabilities = [ns_oc[2] / self.model[state, action][0] for ns_oc in self.model[state, action][3]] # get the probability for each new state/reward as a ratio of the occurence of the new state/reward couple and state/action
        
        choices = range(len(pos_probabilities))
        ns_idx = np.random.choice(choices, p = pos_probabilities)
        new_state = pos_outcomes[ns_idx][0]
        reward = pos_outcomes[ns_idx][1] + self.sim_k * np.sqrt(sa_time)
       
        return new_state, reward
```
This method first evaluate the time elapsed from the last occurence of the given state action, this time difference will be needed in the following evaluation of the bonus reward.
Secondly it collect all the possible state/reward successors couples already encountered for the given state, with the associated probabilities, evaluated as teh ratio of the occurence of the successor state/reward couple and the occurence of the root state.
Finally it randomly pick a new state/reward that is then returned.

Returning to the main method the algorithm contineu with the update of the simulated state/action couple state-action value.
Further all teh state/action couples predicted to lead to the simulated state are evaluated by the `SA_predict()` method.
```
    def SA_predict(self, state):
        predict = []
        for key in self.model.keys():
            for ns in self.model[key][3]:
                if (ns[0] == state):
                    predict.append([key[0], key[1], ns[1]])
        return predict
```
This method, given the state as input, search over the whole model for a state/action couple that led to the desired state, by checking the 4-th component of the model element, and append such state/action together with the experienced reward, to the predict list that is finally returned.

Finally the main method update the priority list by the `` method for all the state/action obtained in the prediction phase.
Thereafter the new experienced state is set as the current one and an if condition check for the end of the episode, done by reaching a terminal state or by reaching a threshold over the number of time steps; reset the time for the timestep threshold and collect the desired data.

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
