import constants
import random
class Drunkard:
    def __init__(self):
        self.q_table= None

    def get_next_action(self,state):

        return constants.Forward if random.random() > 0.5 else constants.Backward
    def update(self, old_state, new_state, action, reward):
        # print("drunk person cant decide")
        pass

class Greedy:
    def __init__(self,length):
        self.q_table=[[0 for i in range(length)] , [0 for i in range(length) ]]
    
    def get_next_action(self, state):
         if self.q_table[constants.Backward][state] > self.q_table[constants.Forward-1][state]:
             return constants.Forward

         elif self.q_table[constants.Backward][state] > self.q_table[constants.Forward-1][state]:
              return constants.Backward
         
         return constants.Forward if random.random() > 0.5 else constants.Backward
    
    def update(self, old_state, new_state, action, reward):
        self.q_table[action][old_state] += reward

class Smart:
    def __init__(self,length,lr=0.2,discount=0.95,exploration_rate=1.0,iterations=10000):
        self.q_table=[[0 for i in range(length)] , [0 for i in range(length) ]]
        self.lr=lr
        self.er=exploration_rate
        # self.iter=iterations
        self.dis=discount
        self.exp_delta=exploration_rate/iterations

    def get_next_action(self,state):
         if random.random()> self.er :
             return self.greedyR(state)   #greedy
         else:
             return self.gambling(state)  #gambel
    def greedyR(self,state):
         if self.q_table[constants.Backward][state] > self.q_table[constants.Forward-1][state]:
             return constants.Forward

         elif self.q_table[constants.Backward][state] > self.q_table[constants.Forward-1][state]:
              return constants.Backward
         
         return constants.Forward if random.random() > 0.5 else constants.Backward
    def gambling(self,state):
        return constants.Forward if random.random() > 0.5 else constants.Backward

    def update(self, old_state, new_state, action, reward):
        old_val=self.q_table[action][old_state]
        future_action=self.greedyR(new_state)
        future_reward=self.q_table[future_action][new_state]
        new_val=old_val + self.lr*(reward+self.dis*future_reward-old_val)
        self.q_table[action][old_state] = new_val

        if self.er > 0:
            self.er -= self.er

