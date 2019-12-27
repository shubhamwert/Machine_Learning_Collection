import gym
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
from constants import *
from DeepQNetwork import DQN
from torch.optim import RMSprop
from replayBuffer import ReplayBuffer
import random,math
from itertools import count
from collections import namedtuple
import torch.nn.functional as F
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




def getcartLocation(env,screen_width):
    world_width=env.x_threshold*2
    scale=screen_width/world_width
    return int(env.state[0] * scale + screen_width / 2.0) #cart centre

def get_screen_size(env):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = getcartLocation(env,screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]
    
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    
    return resize(screen).unsqueeze(0).to(device)
def select_action(policy_net,steps_done,state):
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(memory,policy_net,target_net,optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute  loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env=gym.make('CartPole-v0').unwrapped
    env.reset()

    init_screen = get_screen_size(env)
    _, _, screen_height, screen_width = init_screen.shape


    #network
    n_actions=env.action_space.n
    policy_net=DQN(screen_height,screen_width,n_actions).to(device)
    target_net=DQN(screen_height,screen_width,n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = RMSprop(policy_net.parameters())
    memory = ReplayBuffer(10000)
    steps_done = 0
    episode_durations = []
    

    num_episodes = 50
    for i_episode in range(num_episodes):
     # Initialize the environment and state
     env.reset()
     last_screen = get_screen_size(env)
     current_screen = get_screen_size(env)
     state = current_screen - last_screen
     env.render()
     for t in count():
        # Select and perform an action
        action = select_action(policy_net,steps_done,state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        steps_done+=1
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen_size(env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model(memory,policy_net,target_net,optimizer)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        env.render()
    # Update the target network, copying all weights and biases in DQN
     if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

     print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
