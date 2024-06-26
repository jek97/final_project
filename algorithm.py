#!/usr/bin/env python3
from environment.dungeon import DunegeonEnvironment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict
import math as mth

# Available actions:
UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3

# dyna Q with model and q in a single structure, it also has prioritaized sweeping, let's try to make it stop when Q is stable and see what happend, let's try to create the model while we encounter it, add the dynamic thing of the environment, improve the dictionsary, working but to check, also now we are increasing the model each time we don't find a state action, decide if it's what you want to do or not to keep the algorithm faster
class Algorithm():

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
        self.model = dict() # occurence, time, Q, new states

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
    # verified
    
    def q_update(self, state, action, reward, new_state):
        q = []
        actions = self.env.valid_actions(new_state) # get all the possible actions
        for act in actions: # for all the possible actions
            try:
                q.append(self.model[new_state, act][2]) # if we've encountered it add it to the q list
            except:
                q.append(0) # else init it to zero

        max_q = max(q) # check the maximum one    
        self.model[state, action][2] += self.alpha * (reward + max_q - self.model[state, action][2]) # update the q

    # verified

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
            
    
    # verified

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

        #print("state", state, "action", action, "reward", reward,"new_state", new_state, "p", p)
        #print("model state action", self.model[state, action])
        #print("max q", max_q)
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
    
    # verified

    def simulation(self, state, action, t):
        sa_time = t - self.model[state, action][1] # get the number of timesteps from the last time we tried that state action
        pos_outcomes = [ns_r[0:2] for ns_r in self.model[state, action][3]] # get the possible new state/reward
        pos_probabilities = [ns_oc[2] / self.model[state, action][0] for ns_oc in self.model[state, action][3]] # get the probability for each new state/reward as a ratio of the occurence of the new state/reward couple and state/action
        
        choices = range(len(pos_probabilities))
        ns_idx = np.random.choice(choices, p = pos_probabilities)
        new_state = pos_outcomes[ns_idx][0]
        reward = pos_outcomes[ns_idx][1] + self.sim_k * np.sqrt(sa_time)
       
        return new_state, reward
    
    # verified

    def SA_predict(self, state):
        predict = []
        for key in self.model.keys():
            for ns in self.model[key][3]:
                if (ns[0] == state):
                    predict.append([key[0], key[1], ns[1]])
        return predict
    
    # verified

    def dec_bell_shaped_function(self, max, max_t, s, s_t, t): # max = max of the cos oscillation, s = final value of the curve, max_t = time at which i have the maximum, s_t + max_t = time at which i stabilize at s
        if t < max_t:
            return max + s
        elif max_t <= t <= (s_t + max_t):
            return s + (max/2) * (1 + mth.cos((t - max_t) * (mth.pi/(s_t))))
        else:
            return s
        

    def dyna_q(self):
        # initialization
        time_out = self.rows * self.column
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
                if (reward != 0): print("reward ", reward, "state ", state) 
                
                self.model_update(state, action, t, reward, n_state)
                PQueue = self.pq_update(state, action, reward, n_state, PQueue)
                
                sim_count = -1
                while ((PQueue != []) and (sim_count <= self.n_simulations)):
                    sim_count += 1
                    sim_state, sim_action = PQueue[0][0:2]
                    del PQueue[0]
                        
                    sim_new_state, sim_reward = self.simulation(sim_state, sim_action, t)
                    
                    self.q_update(sim_state, sim_action, sim_reward, sim_new_state)
                    
                    pred = self.SA_predict(sim_state)
                    
                    for pr in pred:
                        PQueue = self.pq_update(pr[0], pr[1], pr[2], sim_state, PQueue)

                
                state = n_state

                if ((timestep.is_last()) or (t >= t_out)): # if we reached the end of the episode or the step limit (to prevent loops)
                    print("episode: ", i, " steps: ", t - old_t, " cum reward: ", cum_rew)
                    self.finish_rt .append(t - old_t)
                    self.cumm_rew.append(cum_rew)
                    t_out = t + time_out
                    old_t = t
                    done = True
        
        return traj
    

    def dyna_q_test(self):
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
            #PQueue = []
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
                #print("\n")
                #print("A state", state, "action", action, "reward", reward, "new state", n_state)

                if (save == True): traj.append([state, action])
                if (reward != 0): print("reward ", reward, "state ", state) 
                
                
                self.model_update(state, action, t, reward, n_state)
                #print("B model state action ",self.model[state, action] )
                
                PQueue = self.pq_update(state, action, reward, n_state, PQueue)
                #print("C PQ", PQueue)
                
                sim_count = 0
                while ((PQueue != []) and (sim_count <= self.n_simulations)):
                    sim_count += 1
                    sim_state, sim_action = PQueue[0][0:2]
                    del PQueue[0]
                    #print("D PQ", PQueue)
                    new_sim_state, sim_reward = self.simulation(sim_state, sim_action, t)
                    #print("sim_state", sim_state, "sim action", sim_action, "new sim state", new_sim_state, "sim reward", sim_reward)
                    #print("model sim B", self.model[sim_state, sim_action])
                    self.q_update(sim_state, sim_action, sim_reward, new_sim_state)
                    #print("model sim A", self.model[sim_state, sim_action])
                    pred = self.SA_predict(sim_state)
                    #print("pred", pred)
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
    
    def policy_eval(self):
        policy = np.ones((self.rows, self.column)) * 5
        for state in self.states:
            q = []
            a_actions = []
            actions = self.env.valid_actions(state) # get all the possible actions
            for act in actions: # for all the possible actions
                try:
                    q.append(self.model[state, act][2]) # if we've encountered it add it to the q list
                    a_actions.append(act)
                except:
                    pass

            try:
                policy[state[0]][state[1]] = a_actions[q.index(max(q))]
            except:
                pass
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
    
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.2, 0.2, 0.001, 50, 10, 0.01)
    traj = solver.dyna_q_test()
    policy = solver.policy_eval()
    solver.plot_arrow_grid(policy, traj, "graph")
    solver.plot_data()
    
    
    
       
    
    
    
if __name__ == "__main__":
    main()

"""
DEBUG
    # eps_greedy_action: 
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.5, 0.5, 0.1, 1, 1,  0.2)
    
    solver.model[(10, 1), 2] = [0, 0, 2, []]
    solver.model[(10, 1), 0] = [0, 0, 4, []]
    solver.model[(10, 1), 1] = [0, 0, 0, []]
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)
    act = solver.eps_greedy_action((10, 1))
    print("act", act)

    
    # q_update: 
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    
    solver.model[(10, 1), 2] = [0, 0, 2, []]
    solver.model[(10, 2), 1] = [0, 0, 3, []]
    print("model_b", solver.model)
    solver.q_update((10, 1), 2, 5, (10, 2))
    print("model_a", solver.model)

    
    # model_update:
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    
    #solver.model[(10, 1), 2][2] = 2
    print("model_1", solver.model)
    solver.model_update((10, 1), 2, 5, 3, (10, 2))
    print("model_2", solver.model)
    solver.model_update((10, 1), 2, 10, 3, (10, 2))
    print("model_3", solver.model)
    solver.model_update((10, 1), 2, 15, 2, (10, 4))
    print("model_3", solver.model)


    # pq_update:
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    pq = []
    print("pq 1", pq)
    solver.model_update((10, 1), 2, 5, 3, (10, 2))
    pq = solver.pq_update((10, 1), 2, 3, (10, 2), pq)
    print("pq 2", pq)
    pq = solver.pq_update((10, 1), 2, 4, (10, 2), pq)
    print("pq 3", pq)

    
    # simulation:
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    
    solver.model[(10, 1), 2][2] = 2
    solver.model_update((10, 1), 2, 3, 10, (12, 2))
    solver.model_update((10, 1), 2, 5, 3, (10, 2))
    solver.model_update((10, 1), 2, 7, 4, (11, 2))
    print("model_3", solver.model)
    new_state, reward = solver.simulation((10, 1), 2, 10)
    print("new_state", new_state, "reward", reward)


    # SA_predict:
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.3, 0.5, 0.1, 1, 1,  0.2)
    
    solver.model[(10, 1), 2][2] = 2
    solver.model_update((10, 1), 2, 3, 10, (12, 2))
    solver.model_update((10, 1), 2, 5, 3, (10, 2))
    solver.model_update((10, 1), 2, 7, 4, (11, 2))
    solver.model_update((18, 4), 2, 10, 4, (10, 2))
    print("model_3", solver.model)
    predict = solver.SA_predict((10, 2))
    print("predict", predict)
    
    # dec_bell_shaped_function
    env = DunegeonEnvironment()
    solver = Algorithm(env, 0.1, 0.2, 0.01, 100, 10, 0.001)
    x = range(100)
    y = []
    for i in x:
        y.append(solver.dec_bell_shaped_function(5, 10, 1, 40, i))
    
    plt.figure()  # Set the size of the figure
    plt.plot(x, y)  # Plot the first functio
    plt.show()

"""