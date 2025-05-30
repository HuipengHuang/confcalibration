import torch
x = torch.ones(size=(32,10))
y = torch.ones(size=(10,))
print(torch.cat((x, y.view(1,-1)), dim=0).shape)