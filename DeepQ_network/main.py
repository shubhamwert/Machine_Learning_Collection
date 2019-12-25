from artifacts import DungeonSimulation
from agents import *
import constants
import time

def run(agent,iterations=1000):
    dungeon=DungeonSimulation()
    dungeon.reset()
    total_reward=0

    for i in range(iterations):
        old_state=dungeon.state
        action = agent.get_next_action(old_state)
        new_state,reward=dungeon.take_action(action)
        agent.update(old_state,new_state,action,reward)

        total_reward +=reward
        if i%10 ==0:
            print("step : ",i," reward : ",total_reward)

        time.sleep(0.0001)
    
    print("Final Q-table", agent.q_table)
    return total_reward
    


if __name__ == "__main__":
    i=10000

    agent=[Smart(5,iterations=i),Greedy(5),Drunkard()]
    reward_list=[0,0,0]
    
    p=0
    for d in agent:
        reward_list[p]= run(d,i)
        p+=1
    print("{} \n{}".format(agent,reward_list))