from tkinter import Variable

import torch
from tensorflow.python.ops import nn

from beginner_source.blitz.cifar10_tutorial import net

output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
