import torch.nn as nn
import torch

class ResNet(nn.Module):
    def __init__(self, blocks_sizes, bootleneck, num_of_classes=200, image_size = 224, in_channels = 3,  ):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_of_classes = num_of_classes
        
        self.blocks_sizes = blocks_sizes
        self.bootleneck = bootleneck
    
        self.output_size = self.image_size // 32

        self.dimension_reduction =  nn.Sequential(
            ConvNormRelu(self.in_channels, 64, 7, 2, 3),
            nn.MaxPool2d(3,2, padding=1))

        self.current_num_of_channels = 64
        self.next_num_of_channels = 64

        self.resnet = nn.ModuleList()

        for block_size in self.blocks_sizes:
            self.make_a_block(block_size)

        self.prediction_head = nn.Sequential(
            nn.AvgPool2d(self.output_size),
            nn.Flatten(),
            nn.Linear(2048 if bootleneck else 512, num_of_classes)
        )


    def make_a_block(self, block_size):
        
        for _ in range(block_size):
            if self.bootleneck:
                #! TO DO: implement Bootleneck dimensions changes
                self.resnet.append(ResNetBootleneckBlock(self.current_num_of_channels,self.next_num_of_channels, self.next_num_of_channels*2))
            else:
                self.resnet.append(ResNetVanillaBlock(self.current_num_of_channels, self.next_num_of_channels))

            if self.current_num_of_channels != self.next_num_of_channels:
                self.current_num_of_channels*=2
        
        self.next_num_of_channels *= 2


    def forward(self, X):
        X = self.dimension_reduction(X)

        for block in self.resnet:
            X = block(X)

        return self.prediction_head(X)


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvNormRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, X):
        return self.block(X)

class NormReluConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(NormReluConv, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride, padding = padding),
        )

    def forward(self, X):
        return self.block(X)

class ResNetVanillaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetVanillaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.first_stride = 2 if self.in_channels != self.out_channels else 1

        self.block = nn.Sequential(
            NormReluConv(in_channels, out_channels, 3, stride=self.first_stride, padding=1),
            NormReluConv(out_channels, out_channels, 3, stride=1, padding=1),
        )

        self.dimension_matcher = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 2),
                                              nn.BatchNorm2d(out_channels)) if self.in_channels != self.out_channels else None

        
    def forward(self, X):
        X_residual = self.block(X)
        #print(X_residual.shape)
        if self.dimension_matcher:
            X = self.dimension_matcher(X)
        #print(X.shape)
        return X_residual + X

class ResNetBootleneckBlock(ResNetVanillaBlock):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(ResNetBootleneckBlock, self).__init__(in_channels, out_channels)

        self.intermediate_channels = intermediate_channels
        self.block = nn.Sequential(
            NormReluConv(in_channels, intermediate_channels, 1, stride=1),
            NormReluConv(intermediate_channels, intermediate_channels, 3, stride=self.first_stride, padding=1),
            NormReluConv(intermediate_channels, out_channels, 1, stride=1),
        )


def get_ResNet18(num_of_classes, input_size, num_of_channels):
    return ResNet([2,2,2,2], False, num_of_classes,input_size, num_of_channels)

if __name__ == '__main__':
    googlenet = ResNet([2,2,2,2],False,200, 224, 3)
    input = torch.randn(1,3,224,224)
    print(googlenet.output_size)
    print(googlenet(input).shape)