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
        self.obstacles = self.obstacles_finder()

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
            actions = self.valid_actions(state)
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
            actions = self.valid_actions(state)
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

    def obstacles_finder(self):
        space = []
        obstacles = []
        for i in range(self.rows):
            for j in range(self.column):
                space.append([i, j])

        for w in range(len(space)):
            found = False
            for k in range(len(self.states)):
                if space[w][0] == self.states[k][0] and space[w][1] == self.states[k][1]:
                    found = True
            if found == False:
                obstacles.append(space[w])

        
        return obstacles

                

    def valid_actions(self, state):
        retval = [UP,DOWN,LEFT,RIGHT]
        obstacles = self.obstacles
        
        if state[0] == 0:  
            retval.remove(UP)
        elif state[0] == self.rows-1:
            retval.remove(DOWN)
            
        if state[1] == 0:  
            retval.remove(LEFT)
        elif state[1] == self.column-1:
            retval.remove(RIGHT)
            
        if [state[0]-1,state[1]] in obstacles:
            retval.remove(UP)
            
        if [state[0]+1,state[1]] in obstacles:
            retval.remove(DOWN)
            
        if [state[0],state[1]+1] in obstacles:
            retval.remove(RIGHT)
            
        if [state[0],state[1]-1] in obstacles:
            retval.remove(LEFT)
            
        
        return retval
           
    
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
    
    
    



"""

    def pos_to_index_and_act(self, M, x, y): # get the index of the cell with associated position x, y and possible actions/q factors
        for i in range(len(M)):
            if M[i][0][0] == x and M[i][0][1] == y:
                return i, M[i][1]
            
        raise Exception("index not found") 
    
    def pos_act_to_index(self, M, x, y, a): # get the index of the cell with associated position x, y and possible actions/q factors
        for i in range(len(M)):
            if M[i][0][0] == x and M[i][0][1] == y:
                for j in range(len(M[i][1])):
                    if M[i][1][j] == a:
                        return i, j
            
        raise Exception("index not found") 
    

    def col_free_map(self): # create a mask with the obstacles signed with a 0 and the free space with a 1
        col_free = np.zeros((self.rows, self.column)) # create an array with the same sizes of the environment
        for i in range(len(self.states)): # for each element that is collision free
            col_free[self.states[i][0]][self.states[i][1]] = 1 # assign a value of 1
        
        return col_free

    def possible_actions(self): # create the possible actions associated to a given state considering the boundaries and the obstacles
        p_a = [] # init the array of possible actions, it will have the shape of [[x, y], [possible actions]]

        for i in range(self.rows): # remove the actions associated only to the boundaries
            for j in range(self.column):
                if i == 0: # first row, you can't go down
                    if j == 0: # also first column so you can't go left neither
                        p_a.append([[i, j], [0, 3]])
                    elif j == self.column-1: # also last column so you can't go right neither
                        p_a.append([[i, j], [0, 2]])
                    else:
                        p_a.append([[i, j], [0, 2, 3]])

                elif i == self.rows-1: # last row, you can't go up
                    if j == 0: # also first column so you can't go left neither
                        p_a.append([[i, j], [1, 3]])
                    elif j == self.column-1: # also last column so you can't go right neither
                        p_a.append([[i, j], [1, 2]])
                    else:
                        p_a.append([[i, j], [1, 2, 3]])

                else:
                    if j == 0: # first column so you can't go left
                        p_a.append([[i, j], [0, 1, 3]])
                    elif j == self.column-1: # last column so you can't go right
                        p_a.append([[i, j], [0, 1, 2]])
                    else:
                        p_a.append([[i, j], [0, 1, 2, 3]])

        col_free = self.col_free_map() # give me the map of obstacles

        for i in range(len(col_free)):
            for j in range(len(col_free[0])): # for all the cells
                if col_free[i][j] == 0: # if the cell is an obstacle

                    try: # remove all the possible actions for that cell
                        [indexx, actions] = self.pos_to_index_and_act(p_a, i, j)
                        p_a[indexx][1] = []
                    except:
                        #print("error")
                        pass

                    try: # try to pick the cell over it and delete the action go down
                        [indexx, actions] = self.pos_to_index_and_act(p_a, i+1, j)
                        p_a[indexx][1].remove(1)
                    except:
                        #print("index out of grid")  
                        pass
                    
                    try: # try to pick the cell under it and delete the action go up
                        [indexx, actions] = self.pos_to_index_and_act(p_a, i-1, j)
                        p_a[indexx][1].remove(0)
                    except:
                        #print("index out of grid") 
                        pass
                    
                    try: # try to pick the cell at its left and delete the action go right
                        [indexx, actions] = self.pos_to_index_and_act(p_a, i, j-1)
                        p_a[indexx][1].remove(3)
                    except:
                        #print("index out of grid")  
                        pass
                    
                    try: # try to pick the cell at its right and delete the action go left
                        [indexx, actions] = self.pos_to_index_and_act(p_a, i, j+1)
                        p_a[indexx][1].remove(2)
                    except:
                        #print("index out of grid") 
                        pass

        return p_a
    
    def q_init(self):
        q_0 = self.possible_actions() # init the q array equal to the possible actions one
        for i in range(len(q_0)): # for all the elements of q
            for j in range(len(q_0[i][1])):
                q_0[i][1][j] = 0 # put a zero instead of the action number
                
        return q_0

    def custom_remove(self, vec, element):
        idx = vec.index(element)
        new_vec = []
        for i in range(len(vec)):
            if i == idx:
                pass
            else:
                new_vec.append(vec[i])

        return new_vec


    def policy_init(self):
        p_0 = self.possible_actions() # init the policy array equal to the possible actions one
        p_a = self.possible_actions() # get the possible actions
        a_actions = [] # init the array of available actions

        for i in range(len(p_0)): # for all the cells
                try:
                    if i == 0: # first cell to be set
                        p_0[i][1] = np.random.choice(p_a[i][1]) # pick randomly an action among the available
                    else: # remove the action that leads to the previous state, otherwise we fall in a infinite loop
                        a_actions = p_a[i][1] # get the current cell
                        a_o_actions = a_actions
                        x = p_a[i][0][0] # cell x position (row)
                        y = p_a[i][0][1] # cell y position (column)
                        
                        try:
                            [down_idx, down_a] = self.pos_to_index_and_act(p_0, x-1, y)
                            if down_a == 0 and down_idx < i: # if the cell below the current one has been already set and it says go up, don't go down
                                a_actions = self.custom_remove(a_actions, 1)
                        except:
                            pass

                        try:
                            [up_idx, up_a] = self.pos_to_index_and_act(p_0, x+1, y)
                            if up_a == 1 and up_idx < i: # if the cell over the current one has been already set and it says go down, don't go up
                                a_actions = self.custom_remove(a_actions, 0)
                        except:
                            pass
                        
                        try:
                            [right_idx, right_a] = self.pos_to_index_and_act(p_0, x, y+1)
                            if right_a == 2 and right_idx < i: # if the cell on the right of the current one has been already set and it says go left, don't go right
                                a_actions = self.custom_remove(a_actions, 3)
                        except:
                            pass
                        
                        try:
                            [left_idx, left_a] = self.pos_to_index_and_act(p_0, x, y-1)
                            if left_a == 3 and left_idx < i: # if the cell on the left of the current one has been already set and it says go right, don't go left
                                a_actions = self.custom_remove(a_actions, 2)
                        except:
                            pass 
                        
                
                        if a_actions == []:
                            p_0[i][1] = np.random.choice(a_o_actions) # pick randomly an action among the available
                        else:
                            p_0[i][1] = np.random.choice(a_actions) # pick randomly an action among the available
                            
                except:
                    p_0[i][1] = [] # this is an obstacle
                    
        return p_0
    
    def eps_policy(self, p, q, p_a, epsilon): # generate an epsilon greedy policy over the q factors
        for i in range(len(p)): # for all the state of the policy
            x = p[i][0][0] # get the x position of the state 
            y = p[i][0][1] # get the y position of the state 
            
            [idx, q_factors] = self.pos_to_index_and_act(q, x, y) # get the q factors associated to that state and the index
            
            if q_factors != []: # it wasn't an obstacles 
                idx_max = np.argmax(q_factors) # get the index with the maximum q factor
                a_max_q = p_a[i][1][idx_max]
                if np.random.uniform() < epsilon: # be greedy on the q factors
                    p[i][1] = a_max_q
                else: # pick another action at random from the remaining ones
                    remained_actions = p_a[i][1]
                    remained_actions = self.custom_remove(remained_actions, a_max_q)
                    p[i][1] = np.random.choice(remained_actions)
            else: # it was an obstacle
                pass
        return p
    
    def eps_action(self, state, q, p_a, epsilon): # generate an epsilon greedy action over the q factors of the given state
        x = state[0] # get the x position of the state 
        y = state[1] # get the y position of the state 
        [idx, q_factors] = self.pos_to_index_and_act(q, x, y) # get the q factors associated to that state and the index
        idx_max = np.argmax(q_factors) # get the index with the maximum q factor

        if np.random.uniform() < epsilon: # be greedy on the q factors
            a = p_a[idx][1][idx_max]
        else: # pick another action at random from the remaining ones
            remained_actions = p_a[idx][1]
            remained_actions = self.custom_remove(remained_actions, p_a[idx][1][idx_max])
            a = np.random.choice(remained_actions)
    
        return a


    def sarsa(self, n_episodes, alpha, epsilon):
        p_as = self.possible_actions()
        qs = self.q_init()
        ps = self.policy_init()

        for i in range(n_episodes):
            done = False
            count = 0            
            
            if i > 0: # if it's the first iteration take the random policy otherwise be epsilon greedy on the q factors
                ps = self.eps_policy(ps, qs, p_as, epsilon) # generate an epsilon greedy policy on the q factors
                
            else:
                pass
            
            s = self.env.reset().observation # observe the initial state
            state = [s[0], s[1]]
        
            [x, action] = self.pos_to_index_and_act(ps, state[0], state[1]) # get the associated action from the policy
        
            while not done:
                count += 1
               
                timestep = self.env.step(action) # execute the action on the environment
                n_s = timestep.observation # check in which state we land
                n_state = [n_s[0], n_s[1]]
                [x, n_action] = self.pos_to_index_and_act(ps, n_state[0], n_state[1]) # get the new action to perform from the policy
                
                
                [o_1, o_2] = self.pos_act_to_index(p_as, state[0], state[1], action) # get the indexes of the old state cell and the index of the action  performed for the q
                [n_1, n_2] = self.pos_act_to_index(p_as, n_state[0], n_state[1], n_action) # get the indexes of the new state cell and the index of the new action for the q
                
                r = timestep.reward # collect the new reward
            
                qs[o_1][1][o_2] += alpha * (r + qs[n_1][1][n_2] - qs[o_1][1][o_2]) # update the q factors

                action = n_action # sweep the new action with the old one
                state = n_state # sweep the new state with the old one

                if timestep.is_last(): # if we reached the end of the episode
                    #print("Goal State Reached!")
                    done = True

                if count > (self.rows * self.column):
                    done = True 
                

        return ps # once done return the policy
    
    def q_learning(self,  n_episodes, alpha, epsilon):
        p_aq = self.possible_actions()
        qq = self.q_init()
        pq = self.policy_init() # init just to have the right structure, to fill in the end part of the algorithm

        for i in range(n_episodes):
            done = False
            count = 0
            s = self.env.reset().observation # observe the initial state
            state = [s[0], s[1]]
        
            while not done:
                count += 1
                action = self.eps_action(state, qq, p_aq, epsilon) # pick action epsilon greedy on the q factors
                timestep = self.env.step(action) # execute the action on the environment
                n_s = timestep.observation # check in which state we land
                n_state = [n_s[0], n_s[1]]
                

                [o_1, o_2] = self.pos_act_to_index(p_aq, state[0], state[1], action) # get the  indexes of the old state cell and the associated action performed for the q
                [n_1, q_ns] = self.pos_to_index_and_act(qq, n_state[0], n_state[1]) # get the index of the new state and the associated q factors of that state
                n_2 = np.argmax(q_ns) # get the index of the bigger q factor for the new state
                r = timestep.reward # collect the new reward
            
                qq[o_1][1][o_2] += alpha * (r + qq[n_1][1][n_2] - qq[o_1][1][o_2]) # update the q factors

                state = n_state # sweep the new state with the old one

                if timestep.is_last(): # if we reached the end of the episode
                    #print("Goal State Reached!")
                    done = True
                
                if count > (self.rows * self.column):
                    done = True 

        # once done evaluate the obtained policy by beeing greedy on the final q factors:
        for i in range(len(pq)): # for all the states of the policy
            x = pq[i][0][0] # get th x coordinate of the state (cell)
            y = pq[i][0][1] # get the y coordinates of the state (cell)
            [idx_1, q] = self.pos_to_index_and_act(qq, x, y) # get the q factors of that state
            try:
                idx_2 = np.argmax(q) # get the index of the bigger q factor for the new state
                pq[i][1] = p_aq[idx_1][1][idx_2] # assign the action to the policy by being greedy on the q factors
            except:
                pass

        return pq # once done return the policy
    
    def n_step_sarsa(self,  n_episodes, alpha, epsilon, n_step):
        
        p_anss = self.possible_actions()
        qnss = self.q_init()
        pnss = self.policy_init()

        for i in range(n_episodes):
            done = False
            tau = 0
            count_t = 0 # init the counter representing the time step
            actions_n = [] # init the vector where i will store the n actions
            states_n = [] # init the vector where i will store the n states
            rewards_n = [] # init the vector where i will store the n rewards

            s = self.env.reset().observation # observe the initial state
            state = [s[0], s[1]]
        
            [x, action] = self.pos_to_index_and_act(pnss, state[0], state[1]) # get the associated action from the policy


            states_n.append(state) # appen the first element of the n states
            actions_n.append(action) # appen the first element of the n actions
            rewards_n.append(0) # store the new reward we obtain                                         
            
            T = float('inf') 
        
            while not done: # this is like for each step

                if count_t < T: # we didn't reach the terminal state yet
                    timestep = self.env.step(action) # execute the action on the environment
                    n_s = timestep.observation # check in which state we land
                    n_state = [n_s[0], n_s[1]]
                    states_n.append(n_state) # store the new state in which we landed       done in two step to use it in the further
                    rewards_n.append(timestep.reward) # store the new reward we obtain
                    if timestep.is_last(): # if St+1 is the terminal state
                        #print("goal reached")
                        T = count_t + 1 # assign the final terminal state step
                    else:
                        [x, n_action] = self.pos_to_index_and_act(pnss, n_state[0], n_state[1]) # get the new action from St+1 from the policy that will be epsilon greedy
                        actions_n.append(n_action) # append the new action in the vector
                        action = n_action
                    
                tau = count_t - n_step + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + n_step, T)): # thanks to this formula i don't need to clear the vector of returns after n steps but check the indexes for the fact that they start in 0
                        G += rewards_n[i] # summ the rewards (since we are not appling any discount)
                    
                    if (tau + n_step) < T: # meaning we didn't reach the terminal state yet in exploration with n_step 
                        [idx_1, idx_2] = self.pos_act_to_index(p_anss, states_n[tau + n_step][0], states_n[tau + n_step][1], actions_n[tau + n_step] ) # get the indexes to obtain the Q values of the last state and action decided
                        G += qnss[idx_1][1][idx_2] # update G
                        

                    # obtain the tau index
                    [idx_11, idx_22] = self.pos_act_to_index(p_anss, states_n[tau][0], states_n[tau][1], actions_n[tau])
                    qnss[idx_11][1][idx_22] += alpha * (G - qnss[idx_11][1][idx_22]) # update Q
                    
                    pnss[idx_11][1] = self.eps_action(states_n[tau], qnss, p_anss, epsilon)
                
                count_t += 1 # time step of the episode
                
                if count_t > self.rows * self.column:
                    done = True
                
                if tau == (T - 1): # i've reached and surpassed the terminal state of n_steps times so i can conclude
                    done = True               


        return pnss # once done return the policy

        

    def plot_arrow_grid(self, rows, columns, policy, title):
        background = np.ones((rows, columns, 3))
        grid = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                [idx, a] = self.pos_to_index_and_act(policy, rows - (i + 1), j)   
                if a == []:
                    background[i][j] = 0.5
                    grid[i][j] = 5
                else:
                    grid[i][j] = a 
    
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

        

def main():
    env = GridEnvironment()
    solver=Algorithms(env)
    p_s = solver.sarsa(2000, 0.3, 0.8)
    p_ql = solver.q_learning(2000, 0.3, 0.5)
    p_nss = solver.n_step_sarsa(2000, 0.3, 0.8, 8)

    print("for each element of the arrays, the first term composed of two component is the cell position [row, column], the second term is the action chosen")
    print("Sarsa policy")
    print(p_s)
    print("Q-learning policy")
    print(p_ql)
    print("n-step Sarsa policy")
    print(p_nss)

    print(" a graph of the obtained policy will be saved on the current folder")

    solver.plot_arrow_grid(solver.rows, solver.column, p_s, 'Sarsa')
    solver.plot_arrow_grid(solver.rows, solver.column, p_ql, 'Q_learning')
    solver.plot_arrow_grid(solver.rows, solver.column, p_nss, 'n_step_Sarsa')


"""
if __name__ == "__main__":
    main()
