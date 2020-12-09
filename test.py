import torch
from tkinter import Variable

from tensorflow.python.ops import nn

from beginner_source.former_torchies.nn_tutorial import net

output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
