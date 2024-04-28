#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Stefano Carpin
EECS269 - Reinforcement Learning
Environment for Final Project
"""

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


# symbolic constants for the actions available; note that not all actions are
# available in every state -- see functions below
UP = 0 
DOWN = 1
LEFT = 2
RIGHT = 3

TREASUREVALUE = 3

FREE = '0'
OBSTACLE = '1'
HOLE = '2'
START = '3'
GOAL = '4'
TREASURE = '5'


ACTION = ["UP","DOWN","LEFT","RIGHT"]  # for printing

"""
A class representing a dungeon environment. An agent
can move around executing the actions defined above. 
"""
class DunegeonEnvironment(py_environment.PyEnvironment):

    
    # creates and instance of the dungeon; no parameters needed
    def __init__(self):
        
        with open('environment.txt') as myhandle:
            lines = myhandle.read().splitlines()
        
        self._parse_environment(lines)
        
        self._actions = [UP,DOWN,LEFT,RIGHT]
        self._state = np.array((2,),dtype = np.int8)
        
        # action is encoded as per the symbolic constants above
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=min(self._actions), maximum=max(self._actions), name='action') 
        # state is the current position on the grid (row,column); positions are 0 indexed
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum = max(self._ROWS,self._COLS)-1, name='state')
        self._state = np.zeros((2,))
        self._episode_ended = False

    # internal function to read environment definition from file
    def _parse_environment(self,lines):
        self._ROWS = len(lines)
        self._COLS = len(lines[0])
        self._HOLES = []
        self._OBSTACLES = []
        self._TREASURES = []
        self._PORTALS = {}
        
        
        row = 0
        for i in lines:
            col = 0
            for j in i:
                if j == START:
                    self._STARTROW = row
                    self._STARTCOL = col
                elif j == GOAL:
                    self._GOALROW = row
                    self._GOALCOL = col
                elif j == OBSTACLE:
                    self._OBSTACLES.append((row,col))
                elif j == HOLE:
                    self._HOLES.append((row,col))
                elif j == TREASURE:
                    self._TREASURES.append((row,col))
                elif j == FREE:
                    pass  ## free space do nothing
                else: # must be a portal
                    if j not in self._PORTALS.keys(): # new portal found
                        self._PORTALS[j] = [(row,col)]
                    else:
                        self._PORTALS[j].append((row,col))
                col +=1
            row += 1
        self._TREASURES_BACKUP = list(self._TREASURES)

    # standard PyEnvironment methods
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    # when reset, the agent always goes back to default position
    def _reset(self):
        self._state[0] = self._STARTROW
        self._state[1] = self._STARTCOL
        self._TREASURES = list(self._TREASURES_BACKUP)
        self._episode_ended = False
        return ts.restart(self._state)
    
    def _entered_portal(self):
        val = (self._state[0],self._state[1])
        for i in self._PORTALS.values():
            if val in i:
                return True
        return False
    
    def _teleport(self):
        val = (self._state[0],self._state[1])
        for i in self._PORTALS.values():
            if val in i:
                if val == i[0]:
                    self._state[0] = i[1][0]
                    self._state[1] = i[1][1]
                else:
                    self._state[0] = i[0][0]
                    self._state[1] = i[0][1]

    # computes transition and rewards 
    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode, so reset the env and start a new episode
            # a new episode.
            return self.reset()

        # compute new state based on action and model
        self._state,_ = self.sample_new_state(self._state,action)
        
        # determin reward and if the episode ended
        if (self._state[0],self._state[1]) in self._HOLES:
            reward = -4
            self._episode_ended = True
        elif (self._state[0],self._state[1]) in self._TREASURES:
            reward = TREASUREVALUE
            self._TREASURES.remove((self._state[0],self._state[1])) # remove treasure from environment
        elif (self._state[0] == self._GOALROW) and (self._state[1] == self._GOALCOL) :
            reward = 1
            self._episode_ended = True
        elif self._entered_portal():
            self._teleport()
            reward = 0
        else:
            reward = 0
    
        
        if self._episode_ended:  # returns time_step of the appropriate type depending on whether episode ended or not
             return ts.termination(self._state, reward)
        else:
             return ts.transition(self._state, reward)


    # additional interface methods for public use
    
    # number of rows in the grid # OK
    def get_num_rows(self): 
        return self._ROWS
    
    # number of columns in the grid # OK
    def get_num_cols(self): 
        return self._COLS
    
    # returns a new state  and reward sampled from the distribution P(s',r|s,a)
    def sample_new_state(self,state,action):
        # first check that the action is valid for the given state
        if not self.valid_action(state,action):
            raise ValueError("Invalid action given")
        
        noise = np.random.choice(a=[-1,0,1],p=[0.25,0.5,0.25])
        
        new_state = np.array(state)
        
        if action == RIGHT:
            new_state[1] += 1
            new_state[0] += noise
        elif action == LEFT:
            new_state[1] -= 1
            new_state[0] += noise
        elif action == UP:
            new_state[0] -=1
            new_state[1] += noise
        elif action == DOWN:
            new_state[0] +=1
            new_state[1] += noise
 
        # collision ? -- do not move
        if (new_state[0],new_state[1]) in self._OBSTACLES: 
            new_state = np.array(state)
            
        # out of bounds? -- do not move    
        if (new_state[0] < 0 ) or (new_state[1] < 0) or (new_state[0] >= self._ROWS) or ( new_state[1] >= self._COLS):
            new_state = np.array(state)
            
        if (new_state[0],new_state[1]) in self._HOLES:
            reward = -4
        elif (new_state[0] == self._GOALROW) and (new_state[1] == self._GOALCOL) :
            reward = 1
        else:
            reward = 0
            
        
        return new_state,reward
            
    
    # returns the actions allowed for the given state (i.e., for state 
    # s it returns the set A(s) represented as a list.
    # state must be given as (row, column) and can be list, array, or tuple.
    # state must be a valid state; unpredictable results occur if wrong states are returned
    def valid_actions(self,state):
        retval = [UP,DOWN,LEFT,RIGHT]
        
        if state[0] == 0:  
            retval.remove(UP)
        elif state[0] == self._ROWS-1:
            retval.remove(DOWN)
            
        if state[1] == 0:  
            retval.remove(LEFT)
        elif state[1] == self._COLS-1:
            retval.remove(RIGHT)
            
        if (state[0]-1,state[1]) in self._OBSTACLES:
            retval.remove(UP)
            
        if (state[0]+1,state[1]) in self._OBSTACLES:
            retval.remove(DOWN)
            
        if (state[0],state[1]+1) in self._OBSTACLES:
            retval.remove(RIGHT)
            
        if (state[0],state[1]-1) in self._OBSTACLES:
            retval.remove(LEFT)
            
        
        return retval
           
                
    # determines if an action is valid for a state
    # action must be given as an integer
    # state must be given as as (row, column) and can be list, array, or tuple
    def valid_action(self,state,action):
        return action in self.valid_actions(state)
    
    # returns a list with all states; each state is represented as (row,col)
    # note: states are not "ordered"
    def all_states(self):
        all_states_list = []
        for i in range(self._ROWS):
            for j in range(self._COLS):
                if not ( (i,j) in self._OBSTACLES ):
                        all_states_list.append((i,j))
                
        return all_states_list


# simple main function to test things out...
def main():
    env = DunegeonEnvironment()  # create an instance of the environemnt 
    
    done = False
    state = env.reset().observation
    print("Starting random policy...")
    
    states = env.all_states()
    
    while not done:
        print("Current state:",state)
        actionset = env.valid_actions(state)
        action = np.random.choice(actionset)
        print("Executing action:",ACTION[action])
        timestep = env.step(action)
        state = timestep.observation
        if timestep.is_last():
            print("Ended Episode!")
            done = True
     

 
if __name__ == "__main__":
    main()
    
        
        
    
    

