import gym
from Models import Agent as Agents
import numpy as np
import pygame
from pygame.locals import *
from random import randint
import pygame
import time
 #apple , player,current length,heigh,width

class Apple:
    x = 0
    y = 0
    step = 44
 
    def __init__(self,x,y):
        self.x = x * self.step
        self.y = y * self.step
 
    def draw(self, surface, image):
        surface.blit(image,(self.x, self.y)) 
 
 
class Player:
    x = [2]
    y = [2]
    step = 44
    direction = 0
    length = 1
    survive_time=0
    updateCountMax = 2
    updateCount = 0
 
    def __init__(self, length):
       self.length = length
       for i in range(0,2000):
           self.x.append(-100)
           self.y.append(-100)
 
      
       self.x[1] = 1*44
       self.x[2] = 2*44
       
    def update(self):
 
        self.updateCount = self.updateCount + 1
        if self.updateCount > self.updateCountMax:
 
            for i in range(self.length-1,0,-1):
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]
 
            if self.direction == 0:
                self.x[0] = self.x[0] + self.step
            if self.direction == 1:
                self.x[0] = self.x[0] - self.step
            if self.direction == 2:
                self.y[0] = self.y[0] - self.step
            if self.direction == 3:
                self.y[0] = self.y[0] + self.step
 
            self.updateCount = 0
            self.survive_time=self.survive_time+1

    def moveRight(self):
        self.direction = 0
 
    def moveLeft(self):
        self.direction = 1
 
    def moveUp(self):
        self.direction = 2
 
    def moveDown(self):
        self.direction = 3 
 
    def draw(self, surface, image):
        for i in range(0,self.length):
            surface.blit(image,(self.x[i],self.y[i])) 
 
class Game:
    def isCollision(self,x1,y1,x2,y2,bsize):
        if x1==x2 and y1==y2:
            return True
        if x1 >= x2 and x1 <= x2 + bsize:
            if y1 >= y2 and y1 <= y2 + bsize:
                return True
        return False
 
class App:
 
    windowWidth = 800
    windowHeight = 600
    player = 0
    apple = 0
    game_state=False
    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = Game()
        self.player = Player(1) 
        self.apple = Apple(5,5)
 
    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
        
        pygame.display.set_caption('Python game')
        self._running = True
        self._image_surf = pygame.image.load("images/snake_body.jpg").convert()
        self._image_surf = pygame.transform.scale(self._image_surf,(20,20))
        self._apple_surf = pygame.image.load("images/food.jpg").convert()
        self._apple_surf = pygame.transform.scale(self._apple_surf,(20,20))
 
    def on_event(self, event):
        if event.type == '^Z':
            self._running = False
 
    def on_loop(self):
        self.player.update()
 
        for i in range(0,self.player.length):
            if self.game.isCollision(self.apple.x,self.apple.y,self.player.x[i], self.player.y[i],self.apple.step):
                self.apple.x = randint(2,9) * self.apple.step
                self.apple.y = randint(2,9) * self.apple.step
                self.player.length = self.player.length + 1
 
 
        for i in range(2,self.player.length):
            if self.game.isCollision(self.player.x[0],self.player.y[0],self.player.x[i], self.player.y[i],40):
                print("You lose! Collision: ")
                print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
                print("x[" + str(i) + "] (" + str(self.player.x[i]) + "," + str(self.player.y[i]) + ")")
                self.game_end()


        if int(self.windowHeight)<=int(self.player.y[i]):
            print(self.player.x[i]," ",self.player.y[i])
            print("you loose!!!")
            self.game_end()
        if self.player.y[i]<-100:
            print(self.player.x[i]," ",self.player.y[i])

            print("you loose!!!")
            self.game_end()
        if self.windowWidth<=self.player.x[i]:
            print(self.player.x[i]," ",self.player.y[i])

            print("you loose!!!")
            self.game_end()
        if self.player.x[i]<-100:
            print("you loose!!!")
            
            print(self.player.x[i]," ",self.player.y[i])
            
            self.game_end()
        

 
        pass
    def game_end(self):
        game_state=True
            
    def on_render(self):

        self._display_surf.fill((0,0,0))
        self.player.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()
 
    # def on_cleanup(self):
    #     pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        print("game startS::")
    def loopit(self,keys):
   
           print(keys)        
           if keys!=0:
            if (keys == 275):
                self.player.moveRight()
 
            if (keys == 276):
                self.player.moveLeft()
 
            if (keys == 273):
                self.player.moveUp()
 
            if (keys == 274):
                self.player.moveDown()
 
            if (keys==27):
                self._running = False
           
           self.on_loop()
           self.on_render()
 
           time.sleep (50.0 / 1000.0)
        
    
  
    def reset(self):
        self.survive_time=0
        self.player=Player(1)
        self.apple=Apple(6,6)
        obs=self.player.length
        self.player.length=1
        obs=[1,self.player.x[-1],self.player.y[-1],self.apple.x,self.apple.y,self.player.length,self.windowHeight,self.windowWidth]
        # self.on_cleanup()
        return obs
    def get_observations(self):
        info="Doing great"
        obs=[1,self.player.x[-1],self.player.y[-1],self.apple.x,self.apple.y,self.player.length,self.windowHeight,self.windowWidth]
        return [obs,self.player.length+self.player.survive_time,self.game_state,info]

def run(actions):
     env=App() 
     env.on_execute()
     brain=Agents(gamma=0.99,epsilon=0,batch_size=64,n_actions=4,input_dims=[8],lr=0.03)
     scores=[]
     eps_history=[]
     n_games=500
     score=0
    #  keyss=[275,274,275,274,274,274]
    #  env.on_execute()
    #  env.loopit(keyss)
     
     n_games=500
     

     for i in range(n_games):
        if i%10 == 0 and i>0:
            avg_score=np.mean(scores[max(0,i-10):(i+1)])
            print('episode ',i,' score ',score,' epsilon %.3f ', brain.epsilon)
        else :
            print('episode ',i ,' score ',score)
        score=0
        eps_history.append(brain.epsilon)
        
        observation=env.reset()
        done=False

        while not done:
            action=brain.choose_action(observation)
            
            observation_, reward , done, info =step(env,actions[action])
            brain.store_transition(observation,action,reward,observation_,done)
            brain.learn()
            observation=observation_

    #     scores.append(score)

    #  x=[i+1 for i in range(n_games)] 
def step(env,action):
    env.loopit(action)
    return env.get_observations()



if __name__ == "__main__" :
    actions=[K_RIGHT,K_LEFT,K_UP,K_DOWN,K_ESCAPE]
    print(actions)
    run(actions)
    

