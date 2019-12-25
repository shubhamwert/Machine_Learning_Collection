from artifacts import DungeonSimulation
from agents import *
import constants
import time
from matplotlib import pyplot as plt


def run(agent,iterations=1000,reward_list=[]):
    dungeon=DungeonSimulation()
    dungeon.reset()
    total_reward=0
    total_reward_list=[]

    for i in range(iterations):
        old_state=dungeon.state
        action = agent.get_next_action(old_state)
        new_state,reward=dungeon.take_action(action)
        agent.update(old_state,new_state,action,reward)
        total_reward +=reward
        if i%10 ==0:
            print("step : ",i," reward : ",total_reward)
            total_reward_list=total_reward_list+[total_reward]
            print(total_reward_list)


        time.sleep(0.0001)
    
    print("Final Q-table", agent.q_table)
    return total_reward,reward_list+[total_reward_list]
    


if __name__ == "__main__":
    i=10000000

    agent=[Smart(5,lr=0.01,exploration_rate=2.0,iterations=i),Greedy(5),Drunkard()]
    reward_list=[0,0,0]
    
    p=0
    r_l=[]
    for d in agent:
        reward_list[p],r_l= run(d,i,r_l)
        p+=1
        print(r_l)

    print("{} \n{}".format(agent,reward_list))
    a=0
    labels=['Smart','Greedy','Drunk']
    for q in r_l:
        plt.plot(range(int(i/10)),q,label=labels[a])
        a+=1
    plt.legend()
    plt.show()
    