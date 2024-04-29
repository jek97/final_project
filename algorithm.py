#!/usr/bin/env python3
from environment.dungeon import DunegeonEnvironment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# Available actions:
UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3

# dyna Q with model and q in a single structure, it also has prioritaized sweeping, let's try to make it stop when Q is stable and see what happend, let's try to create the model while we encounter it, add the dynamic thing of the environment, improve the dictionsary

class Algorithms():

    # init methods
    def __init__(self, env: DunegeonEnvironment, epsilon, alpha, theta, n_ep, n_sim, sim_k):
        # environment informations:
        self.env = env
        self.rows = env.get_num_rows()
        self.column = env.get_num_cols()
        self.states = env.all_states()

        # dyna-q parameters
        self.epsilon = epsilon
        self.alpha = alpha
        self.theta = theta
        self.n_episode = n_ep
        self.n_simulations = n_sim
        self.sim_k = sim_k
        self.model = []

        # plotting informations:
        self.holes = env._HOLES
        self.obstacles = env._OBSTACLES
        self.treasure = env._TREASURES
        self.portals = env._PORTALS
        self.goal_state  = (env._GOALROW, env._GOALCOL)

        # additional data:
        self.start_state = []
        self.finish_rt = []
        self.cumm_rew = []
        

    def model_init(self):
        actions = []
        for state in self.states:
            actions = self.env.valid_actions(state)
            for action in actions:
                self.model.append({"state": state, "action": action,"Q": 0, "occurence": 0, "time": [], "new_states": []})
        
        return
    
    # access methods
    def model_access(self, state, action, new_state, reward, flag):
        for i in range(len(self.model)):
            if (self.model[i]["state"] == state) and (self.model[i]["action"] == action):
                if (flag == 0): # searching only for state/action couple
                    return i
                
                else: # searching only for new_state/reward couple
                    for j in range(len(self.model[i]["new_states"])):
                        if (self.model[i]["new_states"][j][0] == new_state) and (self.model[i]["new_states"][j][1] == reward):
                            return j
        
        return -1
    
    # utilities
    def max_q(self, state):
        max_q = float('-inf')

        for m in self.model: # if i already know some Q consider them for the max
            if (m["state"] == state) and (m["Q"] >= max_q):
                max_q = m["Q"]

        if (max_q == float('-inf')): max_q = 0 # if i don't know none of them, init to 0

        return max_q

    def max_q_occ_a(self, state): # ass max_q but it check only among actions taken
        max_q = float('-inf')
        for m in self.model:
            if (m["state"] == state) and (m["Q"] > max_q) and (m["occurence"] > 0):
                max_q_a = m["action"]
        return max_q_a
    
    def eps_greedy_action(self, state): # choose an epsilon-gredy action over the q factors on a given state
        local_q = []
        actions = []

        actions = self.env.valid_actions(state)
        for i in range(len(actions)):
            #print("act", act)
            idx = self.model_access(state, actions[i], 0, 0, 0)
            if (idx == -1): # there isnt the state action couple
                local_q.append(0)
                actions.append(actions[i])
            else:
                local_q.append(self.model[idx]["Q"])
                actions.append(self.model[idx]["action"])

        """
        for m in self.model: # first get all the q values associated to the current state
            if m["state"] == state:
                local_q.append(m["Q"])
                actions.append(m["action"])
        """       
    
        max_q = max(local_q)
        idx = local_q.index(max_q)
        
        if (np.random.uniform() < self.epsilon): 
            #print("not greedy")
            del actions[idx]
            act = np.random.choice(actions)
        
        else: # be greedy
            #print("greedy")
            act = actions[idx]
        

        return act
    
    def model_update(self, state, action, time, reward, new_state, flag):
        idx = self.model_access(state, action, 0, 0, 0)
        if (idx == -1):
            self.model.append({"state": state, "action": action,"Q": 0, "occurence": 0, "time": time, "new_states": []})
            idx = len(self.model) - 1

        if (flag == 0) or (flag == 2): # only q update (flag = 0) or both model and q update (flag = 2)
            max_q = self.max_q(new_state)
            self.model[idx]["Q"] +=  self.alpha * (reward + max_q - self.model[idx]["Q"])
        
        if (flag == 1) or (flag == 2): # only model update (flag = 1) or both (flag == 2)
            self.model[idx]["occurence"] += 1 # increase the number of times we get in the couple state, action
            if (time != None):
                self.model[idx]["time"] = time # update the last time step at which we encountered the state action combination

            idx_ns = self.model_access(state, action, new_state, reward, 1) # check if we have already encountered the new state/reward combination
            if (idx_ns == -1): # first time we encounter it
                self.model[idx]["new_states"].append([new_state, reward, 1])
            else: # we've already encountered that new state/reward
                self.model[idx]["new_states"][idx_ns][2] += 1 # increase the counter of times we've encountered it
        
        return
    
    def pq_update(self, pq, state, action, new_state, reward):
        idx = self.model_access(state, action, new_state, reward, 0)
        max_q = self.max_q(new_state)
        p = abs(reward + max_q - self.model[idx]["Q"])

        if (p >= self.theta):
            #print("PQ A", pq)
            # check if it's already in the pq list
            
            if (pq != None):
                for pp in pq:
                    #print("X ", (pq[i][0] == state), "y ", (pq[i][1] == action))
                    if (pp[0] == state and pp[1] == action and pp[2] > p):
                        #print("pqp", (pq[i][2] == p))
                        pp[2] = p
                        #print("PQ L1", pq)
                        return pq
            
            pq.append([state, action, p])
            
            pq.sort(reverse = True, key = lambda pq: pq[2])
            #print("PQ L2", pq)

        
        return pq[0:self.n_simulations]

    
    def rand_obs_state_action(self): # also action
        obs_states = []
        obs_actions = []
        flag = False
        for m in self.model:
            if (m["occurence"] > 0):
                n_state = m["state"]  
                for os in obs_states:
                    if n_state == os:
                        flag = True
                if (flag == False):
                    obs_states.append(n_state)
        
        state_idx = np.random.choice(range(len(obs_states)))
        state = obs_states[state_idx]

        for m in self.model:
            if (m["state"] == state) and (m["occurence"] > 0):
                obs_actions.append(m["action"])

        action = np.random.choice(obs_actions)
        return state, action
    
    

    def simulation(self, state, action, t):
        pos_outcomes = []
        pos_probabilities = []
        sa_time = []
        for m in self.model:
            if (m["state"] == state) and (m["action"] == action):
                #print("state", state, "action", action)
                #print("model", m)
                #print("model_time", m["time"])
                sa_time = t - m["time"]
                #print("sa_time", sa_time)
                for ns in m["new_states"]:
                    pos_outcomes.append([ns[0], ns[1]]) # appending the new state/reward couple to the list
                    pos_probabilities.append((ns[2] / m["occurence"])) # appending the probability for each new state/reward couple

        
        choices = range(len(pos_probabilities))
        ns_idx = np.random.choice(choices, p = pos_probabilities)
        #print("choices", pos_outcomes)
        #print("idx", ns_idx)
        new_state = pos_outcomes[ns_idx][0]
        reward = pos_outcomes[ns_idx][1] + self.sim_k * np.sqrt(sa_time)
        
        return new_state, reward
    
    def SA_predict(self, state):
        predict = []
        for m in self.model:
            for ns in m["new_states"]:
                if (ns[0] == state):
                    predict.append([ns[1], m["state"], m["action"]])
        return  predict


    def model_reset(self):
        for m in self.model:
            m["time"] = 0

    def dyna_q(self):
        # initialization
        time_out = self.rows * self.column
        #self.model_init()
        t = 0
        cum_rew = 0
        PQueue = []
        save = False
        
        for i in range(self.n_episode):
            print("episode: ", i, " steps: ", t, " cum reward: ", cum_rew)
            #t = 0
            done = False
            cum_rew = 0
            #self.model_reset()
            traj = []
            if (i == (self.n_episode - 1)): save = True

            s = self.env.reset().observation # observe the initial state
            state = (s[0], s[1])
            self.start_state = state

            while not done:
                t += 1
                
                action = self.eps_greedy_action(state)
                
                if (save == True): traj.append([state, action])
                #if save == True: traj.append(state)

                timestep = self.env.step(action)
                n_s = timestep.observation # check in which state we land
                n_state = (n_s[0], n_s[1])
                
                reward = float(timestep.reward)

                if (reward != 0): print("reward ", reward, "state ", state)
                cum_rew += reward
                
                self.model_update(state, action, t, reward, n_state, 2)
                
                PQueue = self.pq_update(PQueue, state, action, n_state, reward)
                
                for i in range(self.n_simulations):
                    
                    if (i < len(PQueue)):
                        sim_state = PQueue[i][0]
                        sim_action = PQueue[i][1]
                        del PQueue[i]
                    else:
                        sim_state, sim_action = self.rand_obs_state_action()
                        
                    new_sim_state, sim_reward = self.simulation(sim_state, sim_action, t)
                    
                    self.model_update(sim_state, sim_action, None, sim_reward, new_sim_state, 0)
                    
                    pred = self.SA_predict(state)
                    
                    for pr in pred:
                        r = pr[0]
                        PQueue = self.pq_update(PQueue, pr[1], pr[2], state, r)

                
                state = n_state

                if (timestep.is_last()) or (t >= time_out): # if we reached the end of the episode or the step limit (to prevent loops)
                    time_out += t
                    print("model size", len(self.model))
                    self.finish_rt .append(t)
                    self.cumm_rew.append(cum_rew)
                    done = True
        
        return traj
    
    def max_occ(self, state):
        for m in self.model:
            if (m["state"] == state) and (m["occurence"] > 0):
                return True


    def policy_eval(self):
        policy = np.ones((self.rows, self.column)) * 5
        for state in self.states:
            if self.max_occ(state):
                mx_q_a = self.max_q_occ_a(state)
                policy[state[0]][state[1]] = mx_q_a
        
        return policy


    def plot_arrow_grid(self, policy, traj, title):
        rows = len(policy)
        columns = len(policy[0])
        background = np.ones((rows, columns, 3))
        grid = policy
        
        for obs in self.obstacles:
            background[obs[0]][obs[1]] = [0.5, 0.5, 0.5] # gray
            
        for t in self.treasure:
            background[t[0]][t[1]] = [1, 1, 0] # yellow

        for h in self.holes:
            background[h[0]][h[1]] = [0, 0, 0] # black
        
        background[int(self.start_state[0])][int(self.start_state[1])] = [0, 1, 0] # green start stare
        background[int(self.goal_state[0])][int(self.goal_state[1])] = [1, 0, 0]
        
    
        # Create the plot
        fig = plt.figure(figsize=(columns, rows))
        plt.imshow(background, cmap=None, interpolation=None)
        fig.set_facecolor("white")

        # Add arrows/portals to the plot
        for i in range(rows):
            for j in range(columns):
                
                color = 'black'

                if grid[i][j] == 0:  # Up arrow
                    plt.arrow(j, i, 0, -0.2, head_width=0.1, head_length=0.1, fc=color, ec=color)
                elif grid[i][j] == 1:  # Down arrow
                    plt.arrow(j, i, 0, 0.2, head_width=0.1, head_length=0.1, fc=color, ec=color)
                elif grid[i][j] == 2:  # Left arrow
                    plt.arrow(j, i, -0.2, 0, head_width=0.1, head_length=0.1, fc=color, ec=color)
                elif grid[i][j] == 3:  # Right arrow
                    plt.arrow(j, i, 0.2, 0, head_width=0.1, head_length=0.1, fc=color, ec=color)
        
        for tr in traj:
            color = 'red'
            if tr[1] == 0:  # Up arrow
                plt.arrow(tr[0][1], tr[0][0], 0, -0.2, head_width=0.1, head_length=0.1, fc=color, ec=color)
            elif tr[1] == 1:  # Down arrow
                plt.arrow(tr[0][1], tr[0][0], 0, 0.2, head_width=0.1, head_length=0.1, fc=color, ec=color)
            elif tr[1] == 2:  # Left arrow
                plt.arrow(tr[0][1], tr[0][0], -0.2, 0, head_width=0.1, head_length=0.1, fc=color, ec=color)
            elif tr[1] == 3:  # Right arrow
                plt.arrow(tr[0][1], tr[0][0], 0.2, 0, head_width=0.1, head_length=0.1, fc=color, ec=color)
            

        
        port = list(self.portals.values())
        keys = list(self.portals.keys())
        for i in range(len(port)):
            color = np.random.randint(50, 255, (1, 3))
            color = color / 255
            for p in port[i]:
                plt.text(p[1], p[0], keys[i], ha='center', va='center')
                plt.gca().add_patch(Circle((p[1], p[0]), 0.3, color=color))

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

    print("check time steps in the sampling at episode's end")
    env = DunegeonEnvironment()
    solver = Algorithms(env, 0.3, 0.5, 0.01, 10, 5, 0.2)
    traj = solver.dyna_q()
    policy = solver.policy_eval()
    print("model", solver.model)
    solver.plot_arrow_grid(policy, traj, "graph")
    solver.plot_data()
    
    
    
if __name__ == "__main__":
    main()

"""
DEBUG:

    # model init
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    print("model_0")
    print(model)

    # model_access
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    state = model[10]["state"]
    action = model[10]["action"]
    model[10]["new_states"].append([[10, 10], [4], 100])
    model[10]["new_states"].append([[10, 13], [2], 10])
    idx = solver.model_access(model, state, action, [10, 13], [2], 0)
    idx2 = solver.model_access(model, state, action, [10, 13], [2], 1)
    print("idx", idx, idx2)
    print("model 10", model[idx])

    # max_q
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    state = model[10]["state"]
    model[10]["Q"] = 100
    max_q = solver.max_q(model, state)
    print("max_q", max_q)

    # eps_greedy_action: activate the print greedy
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    model[10]["Q"] = 100
    state = model[10]["state"]
    action = model[10]["action"]
    print("state", state, "action", action)
    act = solver.eps_greedy_action(state, model)
    print("act", act)

    # model update
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state, 1)
    print("model_a new state", model[10])
    time = 15
    solver.model_update(model, state, action, time, reward, new_state, 1)
    print("model_a same state", model[10])
    new_state = [5, 10]
    reward = 3
    time = 20
    solver.model_update(model, state, action, time, reward, new_state, 1)
    print("model_a different state", model[10])

    # q update
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    reward = 5
    new_state = [10, 4]
    solver.q_update(model, state, action, 0, reward, new_state, 0)
    print("model_a", model[10])

    # rand_obs_state_action
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state, 1)
    print("model_a new state", model[10])
    st = solver.rand_obs_state_action(model)
    print("previously obs state", st)

    # rand_obs_action
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state, 1)
    print("model_a new state", model[10])
    act = solver.rand_obs_action(model, state)
    print("previously obs action", act)

    # simulation: uncomment the prints
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    model = solver.model_init()
    print("model_b", model[10])
    state = model[10]["state"]
    action = model[10]["action"]
    time = 13
    reward = 5
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state, 1)
    time = 14
    reward = 2
    new_state = [10, 4]
    solver.model_update(model, state, action, time, reward, new_state, 1)
    time = 15
    reward = 7
    new_state = [3, 11]
    solver.model_update(model, state, action, time, reward, new_state, 1)
    solver.model_update(model, state, action, time, reward, new_state, 1)
    print("model_a new state", model[10])
    n_s, n_r = solver.simulation(model, state, action)
    print("new state", n_s, "rewward", n_r)

    # dyna q: uncomment the prints
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    q = solver.dyna_q()
    policy = solver.policy_eval(q)
    solver.plot_arrow_grid(policy, "graph")
    
"""