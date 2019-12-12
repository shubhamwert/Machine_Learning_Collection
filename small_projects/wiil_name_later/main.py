import gym
from Models import Agent as Agents
import numpy as np
# from utils import plotLearning





if __name__ == "__main__":
    env=gym.make('LunarLander-v2')
    brain=Agents(gamma=0.99,epsilon=0,batch_size=64,n_actions=4,input_dims=[8],lr=0.03)
    scores=[]
    eps_history=[]
    n_games=500
    score=0

    for i in range(n_games):
        if i%10 == 0 and i>0:
            avg_score=np.mean(scores[max(0,t-10):(i+1)])
            print('episode ',i,' score ',score,' epsilon %.3f ', brain.epsilon)
        else :
            print('episode ',i ,' score ',score)
        score=0
        eps_history.append(brain.epsilon)
        observation=env.reset()
        done=False

        while not done:
            action=brain.choose_action(observation)
            observation_, reward , done, info =env.step(action)
            brain.store_transition(observation,action,repr,observation_,done)
            brain.learn()
            observation=observation_

        scores.append(score)

    x=[i+1 for i in range(n_games)]
    filename= 'lunar-lander.png'

