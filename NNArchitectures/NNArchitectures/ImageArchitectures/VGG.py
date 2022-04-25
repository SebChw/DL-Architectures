import torch.nn as nn
import torch

class VGGsection(nn.Module):
    def __init__(self, num_of_blocks, previous_num_of_filters, current_num_of_filters):
        super(VGGsection, self).__init__()
        self.num_of_blocks = num_of_blocks

        self.blocks = nn.ModuleList()
        self.kernel_size = 3
        self.conv_stride = 1
        self.conv_padding = 1
        self.pool_size = 2
        self.pool_stride = 2
       
        for i in range(num_of_blocks):
            self.blocks.append( nn.Sequential(
                torch.nn.Conv2d(previous_num_of_filters, current_num_of_filters, self.kernel_size, self.conv_stride, self.conv_padding),
                torch.nn.BatchNorm2d(current_num_of_filters),
                torch.nn.ReLU()
            ))
            previous_num_of_filters = current_num_of_filters # as at first iteration we increase num of filters and then it stays the same
        
        self.blocks.append(torch.nn.MaxPool2d(self.pool_size, self.pool_stride))

    def forward(self, X):
        for block in self.blocks:
            X = block(X)

        return X

class VGG(nn.Module):
    def __init__(self, num_of_classes, num_of_blocks: list, input_size=224, num_of_channels=3):
        super(VGG, self).__init__()
        self.num_of_blocks = num_of_blocks
        self.input_size = input_size
        self.num_of_classes = num_of_classes

        self.calculate_flattened_size(self.input_size)

        self.num_of_previous_filters = [num_of_channels, 64, 128, 256, 512,]
        self.num_of_curr_filters = [64, 128, 256, 512, 512]
        self.conv_blocks = nn.ModuleList()

        for i in range(len(self.num_of_blocks)):
            self.conv_blocks.append(VGGsection(self.num_of_blocks[i], self.num_of_previous_filters[i], self.num_of_curr_filters[i]))

        self.fc_part = nn.Sequential(
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_of_classes)
        )

    def forward(self, X):
        for block in self.conv_blocks:
            X = block(X)
        

        X = torch.flatten(X, start_dim = 1)
        X = self.fc_part(X)

        return X

    def calculate_flattened_size(self, input_size=224):
        #In case of VGG only maxpoolings decrease spatial dimension by a factor of 2 and each VGG has 5 max pooling layers
        self.end_num_of_channels = 64 * 2 ** (len(self.num_of_blocks) - 2)
        #print(len(self.num_of_blocks))
        for i in range(len(self.num_of_blocks)):
            kernel, stride = (2,2)
            input_size = int((input_size - kernel) / stride + 1)
            

        self.flattened_size = input_size ** 2 * self.end_num_of_channels

def get_VGG16(num_of_classes, input_size, num_of_channels):
    return VGG(num_of_classes, [2,2,3,3,3], input_size, num_of_channels)

def get_VGG19(num_of_classes, input_size, num_of_channels):
    return VGG(num_of_classes, [2,2,4,4,4], input_size, num_of_channels)
    

if __name__ == '__main__':
    vgg = get_VGG16(100, 224, 3)
    print(vgg.end_num_of_channels)
    input = torch.randn(1,3,224,224)
    print(vgg(input).shape)