import torch
x = torch.ones(size=(32,10))
y = torch.ones(size=(10,))
print((x@y).shape)