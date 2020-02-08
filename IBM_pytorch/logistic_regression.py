import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader,Dataset

class Data(Dataset):
    
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):      
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
class logistic_regression(nn.Module):
    
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
        
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat
data_set=Data()
model = logistic_regression(1)

model.state_dict() ['linear.weight'].data[0] = torch.tensor([[-5]])
model.state_dict() ['linear.bias'].data[0] = torch.tensor([[-10]])
print("The parameters: ", model.state_dict())

c=torch.nn.BCELoss()
trainloader = DataLoader(dataset = data_set, batch_size = 3)
learning_rate = 2
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def train_model(epochs):
    for epoch in range(epochs):
        for x, y in trainloader:
            yhat = model(x)
            loss = c(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(loss)
            
train_model(100)
yhat = model(data_set.x)
label = yhat > 0.5
print("The accuracy: ", torch.mean((label == data_set.y.type(torch.ByteTensor)).type(torch.float)))
plt.plot(label.detach().numpy(),'rx')
plt.plot(data_set.y,'bo')
plt.show()