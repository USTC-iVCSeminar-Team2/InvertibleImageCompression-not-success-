from turtle import forward
import torch
import torch.nn as nn

class InvertibleBasicBlock(nn.Module):

    def __init__(self, in_channels=1, hidden_channels=8, out_channels=1, kernel_size=3) -> None:
        super(InvertibleBasicBlock, self).__init__()

        self.non_linear = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.PReLU()
        )

    def forward(self, x1, y1):

        y2 = y1
        x2 = x1 + self.non_linear(y1)

        return x2, y2

    def inverse(self, x2, y2):

        y1 = y2
        x1 = x2 - self.non_linear(y2)

        return x1, y1


class BasicRGBInvertibleBlock(nn.Module):

    def __init__(self, num_blocks=3) -> None:
        super(BasicRGBInvertibleBlock, self).__init__()

        self.inn_block = nn.ModuleList()
        for _ in range(num_blocks):
            self.inn_block.append(InvertibleBasicBlock())

    def forward(self, images):

        red1 = images[:,0:1]
        green1 = images[:,1:2]
        blue1 = images[:,2:3]

        red2, green2 = self.inn_block[0](red1, green1)
        blue2 = blue1

        red3, blue3 = self.inn_block[1](red2, blue2)
        green3 = green2

        green4, blue4 = self.inn_block[2](green3, blue3)
        red4 = red3

        return torch.cat((blue4, red4, green4), 1)

    def inverse(self, images):

        blue4 = images[:,0:1]
        red4 = images[:,1:2]
        green4 = images[:,2:3]

        red3 = red4
        green3, blue3 = self.inn_block[2].inverse(green4, blue4)

        green2 = green3
        red2, blue2 = self.inn_block[1].inverse(red3, blue3)

        blue1 = blue2
        red1, green1 = self.inn_block[0].inverse(red2, green2)

        return torch.cat((red1, green1, blue1), 1)

class INN(nn.Module):

    def __init__(self, layer_num=3) -> None:
        super(INN, self).__init__()

        self.layer_num = layer_num
        self.blocks = nn.ModuleList()
        for _ in range(layer_num):
            self.blocks.append(BasicRGBInvertibleBlock())

    def forward(self, inputs):
        for i in range(self.layer_num):
            inputs = self.blocks[i].forward(inputs)
        return inputs

    def inverse(self, inputs):
        for i in range(self.layer_num):
            inputs = self.blocks[self.layer_num-i-1].inverse(inputs)
        return inputs


if __name__ == '__main__':

    # model = InvertibleBasicBlock()
    model = INN()
    # data1 = torch.rand((1, 1, 5, 5))
    # data2 = torch.rand((1, 1, 5, 5))
    # print('1:', data1)
    # # print('2:', data2)
    # data3, data4 = model.forward(data1, data2)
    # # print('3:', data3)
    # # print('4:', data4)
    # data5, data6 = model.inverse(data3, data4)
    # print('5:', data5)
    # # print('6:', data6)
    # print(torch.equal(data5, data1))

    # torch.use_deterministic_algorithms(mode=True)

    data = torch.rand((1, 3, 5, 5))

    data1 = model.forward(data)
    data2 = model.inverse(data1)
    print(data)
    # print(data1)
    print(data2)
    print(torch.equal(data2, data))
