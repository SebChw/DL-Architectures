import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_of_classes = 100, input_size = 227, num_of_channels = 3, conv_norm=True, conv_drop=False, lin_norm=False, lin_drop=True):
        super(AlexNet, self).__init__()
        self.input_size = input_size

        self.kernel_sizes = [11, 5, 3, 3, 3]
        self.num_of_filters = [64, 192, 384, 256, 256]
        self.paddings = [2, 2, 1, 1, 1]
        self.strides = [4, 1, 1, 1, 1]
        self.max_poolings = [(3,2), (3,2), None, None, (3,2)] #(kernel_size, stride)
        
        self.conv_modules = nn.ModuleList()

        self.previous_num_of_channels = num_of_channels
        for i in range(len(self.kernel_sizes)):
            self.conv_modules.append(
                ConvBlock(self.previous_num_of_channels, self.num_of_filters[i], self.kernel_sizes[i], self.strides[i], self.paddings[i], conv_norm, conv_drop)
            )
            self.previous_num_of_channels = self.num_of_filters[i]

            if self.max_poolings[i] is not None:
                kernel, stride = self.max_poolings[i]
                self.conv_modules.append(nn.MaxPool2d(kernel,stride))
        
        self.calculate_flattened_size(input_size)

        self.fully_connected_part = nn.Sequential(
            LinearBlock(self.flattened_size, 4096, lin_norm, lin_drop),
            LinearBlock(4096, 4096, lin_norm, lin_drop),
            nn.Linear(4096, num_of_classes)
        )

        
    
    def calculate_flattened_size(self, input_size=224):
        #Equation is (input_size - kernel_size + 2*padding ) / stride  + 1 and in PyTorch it is floored
        for i in range(len(self.kernel_sizes)):
            input_size = int( ( input_size - self.kernel_sizes[i] + 2*self.paddings[i] )  / self.strides[i] +1) 
            if self.max_poolings[i] is not None:
                kernel, stride = self.max_poolings[i]
                input_size = int((input_size - kernel) / stride + 1)

        self.flattened_size = input_size ** 2 * self.num_of_filters[-1] # last convolutional activtion will has such shape.

    def forward(self, X):
        for conv_module in self.conv_modules:
            X = conv_module(X)

        X = torch.flatten(X, start_dim = 1)
        X = self.fully_connected_part(X)
        
        return X


class Block(nn.Module):
    def __init__(self, use_batchnorm=True, use_dropout=False, dropout_p=0.3):
        super(Block, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_bias = False if self.use_batchnorm else True

        self.linear_part = None
        self.batch_norm = None
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, X):
        X = self.linear_part(X)
        if self.use_batchnorm:
            X = self.batch_norm(X)
        X = self.ReLU(X)
        if self.use_dropout:
            X = self.dropout(X)

        return X

class ConvBlock(Block):
    def __init__(self, in_channels, num_of_filters, filter_size, stride, padding, use_batchnorm=True, use_dropout=False, dropout_p=0.3):
        super(ConvBlock, self).__init__(use_batchnorm, use_dropout, dropout_p)

        self.linear_part = nn.Conv2d(in_channels, num_of_filters, filter_size, stride, padding, bias = self.use_bias)
        self.batch_norm = nn.BatchNorm2d(num_of_filters) if self.use_batchnorm else None
       


class LinearBlock(Block):
    def __init__(self, input_size, output_size, use_batchnorm=True, use_dropout=False, dropout_p=0.3):
        super(LinearBlock, self).__init__(use_batchnorm, use_dropout, dropout_p)

        self.linear_part = nn.Linear(input_size, output_size, bias=self.use_bias)
        self.batch_norm = nn.BatchNorm1d(output_size) if self.use_batchnorm else None
        


if __name__ == '__main__':
    net = AlexNet()
    print(net.calculate_flattened_size(64))
    print(net.flattened_size)
    x = torch.randn(1,3,227,227)
    output = net(x)
    print(output.shape)

