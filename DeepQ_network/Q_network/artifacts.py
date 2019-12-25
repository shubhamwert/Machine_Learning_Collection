from constants import *
import random

class DungeonSimulation:
    def __init__(self,length=5,slip=0.1,small=0.2,large=10):
        self.length=length  #total tiles
        self.slip=slip      #prob of slipping action
        self.large=large    #last tile reward
        self.small=small    #backward reward
        self.state=0   #inital state

    def take_action(self,action):
        reward=0
        if random.random() < self.slip:    #random action
            action=not action
        if action == Backward:                     #backward
            reward=self.small
        if action == Forward:
            
            if self.state < self.length -1:
                self.state +=1
                reward=0
            else:
                reward == self.large            #forward
        
        return self.state,reward
    def reset(self):
        self.state=0
        return self.state

