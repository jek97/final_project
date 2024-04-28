#!/usr/bin/env python3
from environment.dungeon import DunegeonEnvironment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Available actions:
UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3


class Algorithms():

    # init methods
    def __init__(self, env: DunegeonEnvironment, epsilon, alpha, n_ep, n_sim):
        # initialize the environment parameters
        self.env = env
        self.rows = env.get_num_rows()
        self.column = env.get_num_cols()
        self.states = env.all_states()
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_episode = n_ep
        self.n_simulations = n_sim

        # additional data
        self.start_state = []
        self.goal_state  = []
        self.finish_rt = []
        self.finish_vt = []
        self.cumm_rew = []
        

    def model_init(self):
        model = []
        state = []
        actions = []
        for i in range(len(self.states)):
            state = [self.states[i][0], self.states[i][1]]
            actions = self.env.valid_actions(state)
            for j in range(len(actions)):
                model.append({"state": state, "action": actions[j], "occurence": 0, "time": [], "new_states": []})
        
        return model
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    model = solver.model_init()
    print("model_0")
    print(model)
    """
    
    def q_init(self):
        q = []
        state = []
        actions = []
        for i in range(len(self.states)):
            state = [self.states[i][0], self.states[i][1]]
            actions = self.env.valid_actions(state)
            for j in range(len(actions)):
                q.append({"state": state, "action": actions[j], "Q": 0})
        
        return q
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    q = solver.q_init()
    print("q_0")
    print(q)
    """


           
    
    # access methods
    def Q_access(self, q, state, action):
        for i in range(len(q)):
            if q[i]["state"] == state and q[i]["action"] == action:
                return i
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    q = solver.q_init()
    q[5]["Q"] = 3
    state = q[5]["state"] 
    action = q[5]["action"]
    print("state", state, "action", action)
    idx = solver.Q_access(q, state, action)
    print("q(idx)", q[idx])
    """
    
    def model_access(self, model, state, action):
        for i in range(len(model)):
            if model[i]["state"] == state and model[i]["action"] == action:
                return i
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    model = solver.model_init()
    state = model[10]["state"]
    action = model[10]["action"]
    idx = solver.model_access(model, state, action)
    print("idx", idx)
    """
            
    def model_ns_access(self, model, state, action, new_state, reward):
        for i in range(len(model)):
            if model[i]["state"] == state and model[i]["action"] == action:
                for j in range(len(model[i]["new_states"])):
                    if model[i]["new_states"][j][0] == new_state and model[i]["new_states"][j][1] == reward:
                        return j
        #print("state/reward combination not found")
        return -1
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    model = solver.model_init()
    state = model[10]["state"]
    action = model[10]["action"]
    model[10]["new_states"].append([[10, 10], [4], 100])
    idx = solver.model_ns_access(model, state, action, [10, 10], [4])
    print("idx", idx)
    """
    
    # utilities
    def max_q(self, q, state):
        max_q = 0
        max_q_idx = 0
        max_q_a = 0
        for i in range(len(q)):
            if q[i]["state"] == state:
                if q[i]["Q"] >= max_q:
                    max_q = q[i]["Q"]
                    max_q_idx = i
                    max_q_a = q[i]["action"]
        return max_q, max_q_idx, max_q_a
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    q = solver.q_init()
    state = q[10]["state"]
    q[10]["Q"] = 100
    max_q, max_q_idx, max_q_a = solver.max_q(q, state)
    print("max_q", max_q, "idx", max_q_idx)
    """
    
    def eps_greedy_action(self, state, q): # choose an epsilon-gredy action over the q factors on a given state
        local_q = []
        actions = []
        
        for i in range(len(q)): # first get all the q values associated to the current state
            if q[i]["state"] == state:
                local_q.append(q[i]["Q"])
                actions.append(q[i]["action"])
        
        max_q = max(local_q)
        idx = local_q.index(max_q)
        
        if np.random.uniform() < self.epsilon: 
            del actions[idx]
            act = np.random.choice(actions)
        
        else: # be greedy
            act = actions[idx]
            
        
        return act
    """
    # main test: activate the print greedy
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    q = solver.q_init()
    q[10]["Q"] = 100
    state = q[10]["state"]
    action = q[10]["action"]
    print("state", state, "action", action)
    act = solver.eps_greedy_action(state, q)
    print("act", act)
    """
    
    def model_update(self, model, state, action, time, reward, new_state):
        idx = self.model_access(model, state, action)
        
        model[idx]["occurence"] += 1 # increase the number of times we get in the couple state, action
        model[idx]["time"] = time # update the last time step at which we encountered the state action combination
        idx_ns = self.model_ns_access(model, state, action, new_state, reward) # check if we have already encountered the new state/reward combination
        if idx_ns == -1: # first time we encounter it
            model[idx]["new_states"].append([new_state, reward, 1])
        else: # we've already encountered that new state/reward
            model[idx]["new_states"][idx_ns][2] += 1 # increase the counter of times we've encountered it
        
        return model
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state)
    print("model_a new state", model[10])
    time = 15
    solver.model_update(model, state, action, time, reward, new_state)
    print("model_a old state", model[10])
    """

    def q_update(self, q, state, action, new_state, reward):
        idx = self.Q_access(q, state, action)
        max_q, mx_q_idx, mx_q_a = self.max_q(q, new_state)
        q[idx]["Q"] = q[idx]["Q"] + self.alpha * (reward + max_q - q[idx]["Q"])

        return q

    
    def rand_obs_state(self, model):
        obs_states = []
        flag = False
        for i in range(len(model)):
            if model[i]["occurence"] > 0:
                n_state = model[i]["state"]  
                for j in range(len(obs_states)):
                    if n_state == obs_states[j]:
                        flag = True
                if flag == False:
                    obs_states.append(n_state)
        
        state_idx = np.random.choice(range(len(obs_states)))
        state = obs_states[state_idx]
        return state
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state)
    print("model_a new state", model[10])
    st = solver.rand_obs_state(model)
    print("previously obs state", st)
    """
    
    def rand_obs_action(self, model, state):
        obs_actions = []
        for i in range(len(model)):
            if model[i]["state"] == state and model[i]["occurence"] > 0:
                obs_actions.append(model[i]["action"])

        action = np.random.choice(obs_actions)
        return action
    
    """
    #main test
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state)
    print("model_a new state", model[10])
    act = solver.rand_obs_action(model, state)
    print("previously obs action", act)
    """

    def simulation(self, model, state, action):
        pos_outcomes = []
        pos_probabilities = []
        for i in range(len(model)):
            if model[i]["state"] == state and model[i]["action"] == action:
                for j in range(len(model[i]["new_states"])):
                    pos_outcomes.append([model[i]["new_states"][j][0], model[i]["new_states"][j][1]]) # appending the new state/reward couple to the list
                    pos_probabilities.append((model[i]["new_states"][j][2] / model[i]["occurence"])) # appending the probability for each new state/reward couple
        
        choices = range(len(pos_probabilities))
        ns_idx = np.random.choice(choices, p = pos_probabilities)
        #print("choices", pos_outcomes)
        #print("idx", ns_idx)
        new_state = pos_outcomes[ns_idx][0]
        reward = pos_outcomes[ns_idx][1]
        
        return new_state, reward
    """
    #main test: uncomment the prints
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state)
    time = 14
    reward = 2
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state)
    time = 15
    reward = 7
    new_state = [3, 11]
    solver.model_update(model, state, action, time, reward, new_state)
    solver.model_update(model, state, action, time, reward, new_state)
    print("model_a new state", model[10])
    n_s, n_r = solver.simulation(model, state, action)
    print("new state", n_s, "rewward", n_r)
    """

    def model_reset(self, model):
        for i in range(len(model)):
            model[i]["time"] = []

    def dyna_q(self):
        # initialization
        q = self.q_init()
        model = self.model_init()
        for i in range(self.n_episode):
            print("episode", i)
            t = 0
            tv = 0
            done = False
            cum_rew = 0
            self.model_reset(model)

            s = self.env.reset().observation # observe the initial state
            state = [s[0], s[1]]
            self.start_state = state

            while not done:
                t += 1
                
                action = self.eps_greedy_action(state, q)
                #print("state", state)
                #print("action", action)
                #for i in range(len(q)):
                    #if q[i]["state"] == state:
                        #print("actions for state", q[i]["action"])
                #print("valid_actions env", self.env.valid_actions(state))
                #print("valid_actions", self.valid_actions(state))

                timestep = self.env.step(action)
                n_s = timestep.observation # check in which state we land
                n_state = [n_s[0], n_s[1]]
                reward = timestep.reward
                cum_rew += reward

                q = self.q_update(q, state, action, n_state, reward)

                model = self.model_update(model, state, action, t, reward, n_state)
                
                for i in range(self.n_simulations):
                    tv += 1
                    sim_state = self.rand_obs_state(model)
                    sim_action = self.rand_obs_action(model, sim_state)
                    new_sim_state, sim_reward = self.simulation(model, sim_state, sim_action)
                    q = self.q_update(q, sim_state, sim_action, new_sim_state, sim_reward)
                    self.finish_vt

                
                state = n_state

                if timestep.is_last(): # if we reached the end of the episode
                    self.finish_rt .append(t)
                    self.finish_vt.append(tv)
                    self.cumm_rew.append(cum_rew)
                    done = True
        
        return q
    
    """
    #main test: uncomment the prints
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 1, 1)
    q = solver.dyna_q()
    policy = solver.policy_eval(q)
    solver.plot_arrow_grid(policy, "graph")
    """


    def policy_eval(self, q):
        policy = np.ones((self.rows, self.column)) * 5
        for i in range(len(self.states)):
            mx_q, mx_q_idx, mx_q_a = self.max_q(q, [self.states[i][0], self.states[i][0]])
            policy[self.states[i][0]][self.states[i][1]] = mx_q_a
        
        return policy


    def plot_arrow_grid(self, policy, title):
        rows = len(policy)
        columns = len(policy[0])
        background = np.ones((rows, columns, 3))
        grid = policy

        for i in range(rows):
            for j in range(columns):
                if policy[i][j] == 5:
                    background[i][j] = 0.5
                if [i, j] == self.start_state:
                    background[i][j] = 0.8  
                if [i, j] == self.goal_state:
                    background[i][j] = 0.2
    
        # Create the plot
        fig = plt.figure(figsize=(columns, rows))
        plt.imshow(background, cmap=None, interpolation=None)
        fig.set_facecolor("white")

        # Add arrows to the plot
        for i in range(rows):
            for j in range(columns):
                if grid[i][j] == 0:  # Up arrow
                    plt.arrow(j, i, 0, -0.2, head_width=0.1, head_length=0.1, fc='black', ec='black')
                elif grid[i][j] == 1:  # Down arrow
                    plt.arrow(j, i, 0, 0.2, head_width=0.1, head_length=0.1, fc='black', ec='black')
                elif grid[i][j] == 2:  # Left arrow
                    plt.arrow(j, i, -0.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
                elif grid[i][j] == 3:  # Right arrow
                    plt.arrow(j, i, 0.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
                

        plt.xlim(-0.5, columns-0.5)
        plt.ylim(rows-0.5, -0.5)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title(title)
        #plt.show(block=False)  
        file_type = ".png"
        graph_name = title + file_type
        plt.savefig(graph_name) 

    def plot_data(self):
        print("cumm reward ", self.cumm_rew)

        # Plot the functions
        plt.figure(figsize=(8, 6))  # Set the size of the figure

        plt.plot(range(len(self.cumm_rew)), self.cumm_rew, label='cumulative reward')  # Plot the first function
        plt.plot(range(len(self.finish_rt)), self.finish_rt, label='real time')  # Plot the second function

        plt.xlabel('episodes')  # Label for x-axis
        plt.ylabel('y')  # Label for y-axis
        plt.title('Plot of cumulative reward = f(episodes) and episede steps = f(episodes)')  # Title of the plot

        plt.legend()  # Display legend

        plt.grid(True)  # Add grid
        plt.show()  # Show the plot
   


def main():
    
    #main test: uncomment the prints
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.1, 0.5, 2000, 8)
    q = solver.dyna_q()
    policy = solver.policy_eval(q)
    print("policy", policy)
    solver.plot_arrow_grid(policy, "graph")
    solver.plot_data()
    
    
if __name__ == "__main__":
    main()
