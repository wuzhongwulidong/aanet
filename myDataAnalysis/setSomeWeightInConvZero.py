import torch
import torch.nn as nn
import torch.nn.functional as F


class diagConv2d_p(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(diagConv2d_p, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding, dilation, groups, bias)

        self.eyeTenor = torch.eye(kernel_size)  # 创建对角矩阵n*n
        self.eyeTenor.requires_grad = False

        self.conv1.weight.data = self.conv1.weight.data * self.eyeTenor
        self.conv1.weight.register_hook(self.tensor_hook)  # tensor_hook会接受到的参数维度为：[输出通道，输入通道，Kh，Kw]

    def tensor_hook(self, grad):
        grad *= self.eyeTenor
        return grad

    def forward(self, x):
        assert self.conv1.weight.data[0, 0, 0, 1] == 0  # 确认只有主对角线上的权重非0，其他权重都为0
        x = self.conv1(x)
        return x


class diagConv2d_n(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(diagConv2d_n, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding, dilation, groups, bias)

        self.eyeTenor = torch.eye(kernel_size).flip([0]).contiguous()  # 创建对角矩阵n*n
        self.eyeTenor.requires_grad = False

        self.conv1.weight.data = self.conv1.weight.data * self.eyeTenor
        self.conv1.weight.register_hook(self.tensor_hook)  # tensor_hook会接受到的参数维度为：[输出通道，输入通道，Kh，Kw]

    def tensor_hook(self, grad):
        grad *= self.eyeTenor
        return grad

    def forward(self, x):
        assert self.conv1.weight.data[0, 0, 0, 0] == 0  # 确认只有反对角线上的权重非0，其他权重都为0.
        x = self.conv1(x)
        return x


# 创建一个很简单的网络：两个卷积层，一个全连接层
class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = diagConv2d_p(3, 4, 3, 1, padding=1, dilation=1, bias=True)
        self.conv1 = diagConv2d_n(1, 1, 3, 1, padding=1, bias=False)

    def forward(self, x):
        # print("conv1.weight.shape:{}".format(self.conv1.weight.data[0, :, :, :]))
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.linear(x.view(x.size(0), -1))
        return x


# # 创建一个很简单的网络：两个卷积层，一个全连接层
# class Simple(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 4, 3, 1, padding=1, bias=True)
#         # self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
#         # self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1, bias=False)
#         # self.linear = nn.Linear(32 * 10 * 10, 20, bias=False)
#
#         # The hook will be called every time the gradients with respect to module inputs are computed.
#         # self.conv1.register_backward_hook(self.my_hook)
#         # self.conv1.register_forward_hook()
#         # self.conv1.weight.register_hook(self.tensor_hook)  # tensor_hook会接受到的参数维度为：[输出通道，输入通道，Kh，Kw]
#
#         # self.conv1.bias.register_hook(self.tensor_hook)
#         kernel_size = 3
#         self.eyeTenor = torch.eye(kernel_size) # 创建对角矩阵n*n
#         self.eyeTenor.requires_grad = False
#
#         self.conv1.weight.data = self.conv1.weight.data * self.eyeTenor
#         self.conv1.weight.register_hook(self.tensor_hook)  # tensor_hook会接受到的参数维度为：[输出通道，输入通道，Kh，Kw]
#
#     def tensor_hook(self, grad):
#         # print('doing tensor_hook')
#         # print('original grad:', grad)
#         grad *= self.eyeTenor
#
#         # grad_input = grad_input[0]*self.input   # 这里把hook函数内对grad_input的操作进行了注释，
#         # grad_input = tuple([grad_input])        # 返回的grad_input必须是tuple，所以我们进行了tuple包装。
#         # print('now grad:', grad_input)
#
#         return grad
#
#
#     def my_hook(self, module, grad_input, grad_output):
#         """
#          grad_input: 对卷积权重的导数
#          grad_output：对数据的导数。
#         """
#         # print('doing my_hook')
#         # print('original grad:', grad_input)
#         # print('original outgrad:', grad_output)
#         # grad_input = grad_input[0]*self.input   # 这里把hook函数内对grad_input的操作进行了注释，
#         # grad_input = tuple([grad_input])        # 返回的grad_input必须是tuple，所以我们进行了tuple包装。
#         # print('now grad:', grad_input)
#
#         return grad_input
#
#     def forward(self, x):
#         assert self.conv1.weight.data[0, 0, 0, 1] == 0
#
#         # print("conv1.weight.shape:{}".format(self.conv1.weight.data[0, :, :, :]))
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         # x = self.linear(x.view(x.size(0), -1))
#         return x


model = Simple()
# 为了方便观察数据变化，把所有网络参数都初始化为 0.1
# for m in model.parameters():
#     m.data.fill_(0.1)

# criterion = nn.CrossEntropyLoss()
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

model.train()
# 模拟输入8个 sample，每个的大小是 10x10，
# 值都初始化为1，让每次输出结果都固定，方便观察
images = torch.ones(2, 1, 10, 10)
targets = torch.ones(2, 1, 10, 10, dtype=torch.long)

iterNum = 10000
for i in range(iterNum):

    output = model(images)
    print(output.shape)
    # torch.Size([8, 20])

    loss = criterion(output, targets)
    print('>>>>>>>>>>>>>>>>>>>>>>>Iteration:{}'.format(i))
    print('>>>>>>>>>>>>>>>>>>>>>>>loss:{}'.format(loss))

    # print('model.conv1.weight.grad:{}'.format(model.conv1.weight.grad))
    # None
    if i == iterNum - 1:
        print(model.conv1.conv1.weight.data)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()





# print('model.conv1.weight.grad.shape:{}'.format(model.conv1.weight.grad.shape))
# print('model.conv1.weight.grad[0][0][0]:{}'.format(model.conv1.weight.grad[0][0][0]))
# tensor([-0.0782, -0.0842, -0.0782])
# 通过一次反向传播，计算出网络参数的导数，
# 因为篇幅原因，我们只观察一小部分结果

# print(model.conv1.weight[0][0][0])
# tensor([0.1000, 0.1000, 0.1000], grad_fn=<SelectBackward>)
# 我们知道网络参数的值一开始都初始化为 0.1 的

optimizer.step()
# print(model.conv1.weight[0][0][0])
# tensor([0.1782, 0.1842, 0.1782], grad_fn=<SelectBackward>)
# 回想刚才我们设置 learning rate 为 1，这样，
# 更新后的结果，正好是 (原始权重 - 求导结果) ！

# optimizer.zero_grad()
# print(model.conv1.weight.grad[0][0][0])
# tensor([0., 0., 0.])
# 每次更新完权重之后，我们记得要把导数清零啊，
# 不然下次会得到一个和上次计算一起累加的结果。
# 当然，zero_grad() 的位置，可以放到前边去，
# 只要保证在计算导数前，参数的导数是清零的就好。