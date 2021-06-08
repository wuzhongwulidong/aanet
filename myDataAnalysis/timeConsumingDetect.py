from torch.autograd import Variable
import torch

x = Variable(torch.randn(1, 1), requires_grad=True)
with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, with_stack=True) as prof:
     y = x ** 2
     y.backward()
# NOTE: some columns were removed for brevity
print(prof)