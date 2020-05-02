import torch 
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn,optim



class withPytorch(torch.nn.Module):
    def __init__(self,input_size,output_size):
        super(withPytorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.creterionLoss=torch.nn.MSELoss()

    def forward(self, x):
        yhat = self.linear(x)
        return yhat
    def loss(self,y,yhat):
        return torch.mean((y-yhat)**2)
    


class basiclr1DModel():
    
    def __init__(self,lr=0.1):
             self.w=torch.randn(1).requires_grad_()
             self.b=torch.randn(1).requires_grad_()
             self.lr=lr
    def forward(self, x):
        yhat=self.w*x + self.b
        return yhat

    def loss(self,yhat,y):
        return torch.mean((yhat-y)**2)
    def fit(self,x,y):
        print('.',end='')
        yhat=self.forward(x)
        
        closs=self.loss(yhat,y)
        closs.backward()
        self.w.data=self.w.data-self.lr*self.w.grad.data
        self.b.data=self.b.data-self.lr*self.b.grad.data
        self.w.grad.data.zero_()
        self.b.grad.data.zero_()
        return closs

class StochasticModel():
    def __init__(self,lr=0.1):
             self.w=torch.randn(1).requires_grad_()
             self.b=torch.randn(1).requires_grad_()
             self.lr=lr
    def forward(self, x):
        yhat=self.w*x + self.b
        return yhat

    def loss(self,yhat,y):
        return torch.mean((yhat-y)**2)
    def fit(self,loader,iterations=10,lossList=[]):
      
      for i in range(iterations): 
       print('.',end='')
       localLoss=[] 
       for x,y in loader:
        yhat=self.forward(x)
        
        closs=self.loss(yhat,y)
        closs.backward()
        self.w.data=self.w.data-self.lr*self.w.grad.data
        self.b.data=self.b.data-self.lr*self.b.grad.data
        self.w.grad.data.zero_()
        self.b.grad.data.zero_()
        localLoss.append(closs)
       lossList = torch.mean(torch.tensor(localLoss))
       
    
      return lossList 


class Data(Dataset):
    def __init__(self,x,y):
        # self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        # self.y = 1 * self.x - 1
        self.x=x
        self.y=y
        self.len = x.shape[0]
        
    # Getter
    def __getitem__(self,index):    
        return self.x[index], self.y[index]
    
    # Return the length
    def __len__(self):
        return self.len
        

def createData():
    X=torch.arange(-3,3,0.02).view(-1,1)
    f=0.3*X+3
    
    Y=f+torch.randn(X.size())*torch.randn(X.size())*0.2
    plt.plot(X.numpy(),f.numpy(),label='f')
    plt.plot(X.numpy(),Y.numpy(),'rx',label='y')
    plt.show()
    return [X,Y]

def train_model_BGD(model,optimizer,trainloader,iter=10):
    for epoch in range(iter):
        for x,y in trainloader:
            yhat = model(x)
            loss = model.loss(yhat, y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
if __name__ == "__main__":
    [X,Y]=createData()
    print("starting learning")
    model=basiclr1DModel()
    loss=[]
    iterations=20
    print('training.')
    for i in range(iterations):
        closs=model.fit(X,Y)
        loss.append(closs)

    #stochastic one
    print("loading data for round 2")
    data=Data(X,Y)
    dataloader=DataLoader(dataset=data,batch_size=1)
    dataloader2=DataLoader(dataset=data,batch_size=5)
    print("running with batch size 1")

    stochasticModel1=StochasticModel()
    l1=stochasticModel1.fit(dataloader,iterations)
    print("running with batch size 10")

    stochasticModel2=StochasticModel()
    lr2=[]
    for i in dataloader2:
        print(".",end='')
        p=stochasticModel2.fit(dataloader2,iterations=2,lossList=lr2)
        lr2.append(p)

    plt.plot(range(iterations),loss,label='whole data')
    plt.plot(l1,label='batch size 1')
    plt.plot(lr2,label='batch size 10')

    plt.show()
    plt.plot(X,Y,'rx',label='original')
    Yhat=model.forward(X)
    plt.plot(X,Yhat.detach().numpy(),label='predicted batch')
    Yhat=stochasticModel1.forward(X)
    plt.plot(X,Yhat.detach().numpy(),label='1 batch')
    Yhat=stochasticModel2.forward(X)
    
    plt.plot(X,Yhat.detach().numpy(),label='10 batch')

    plt.show()


    #pytorch way
    model=withPytorch(1,1)
    optimizer = optim.SGD(model.parameters(), lr = 0.01)

    dataloader=DataLoader(dataset=data,batch_size=1)
    model.state_dict()['linear.weight'][0] = -15
    model.state_dict()['linear.bias'][0] = -10
    train_model_BGD(model,optimizer=optimizer,trainloader=dataloader)
    Yhat=model.forward(X)
    plt.plot(X,Y,'rx',label='original')
    
    plt.plot(X,Yhat.detach().numpy(),label='pytorch one')
    plt.show()
