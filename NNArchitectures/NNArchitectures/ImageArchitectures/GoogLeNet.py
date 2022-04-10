import torch.nn as nn
import torch


class InceptionModule(nn.Module):
    def __init__(self, in_channels, onex1, threex3_reduce, threex3, fivex5_reduce, fivex5, pool):
        super(InceptionModule, self).__init__()
        #All the filters leave the spatial size as it is so we can easily concatenate them along the filter channel, as 
        #Every other dimension will match
        self.one_by_one_block = ConvNormRelu(in_channels, onex1, kernel_size=1, stride=1)

        self.three_by_three_block = nn.Sequential(
            ConvNormRelu(in_channels, threex3_reduce, kernel_size=1, stride = 1),
            ConvNormRelu(threex3_reduce, threex3, kernel_size = 3, stride = 1 , padding = 1)
        )

        self.five_by_five_block = nn.Sequential(
            ConvNormRelu(in_channels, fivex5_reduce, kernel_size=1, stride=1, padding=0),
            ConvNormRelu(fivex5_reduce, fivex5, kernel_size=3, stride=1, padding=1),
        )

        self.pool_block = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvNormRelu(in_channels, pool, kernel_size=1, stride=1, padding=0)
        )

    def forward(self,X):
        X_1 = self.one_by_one_block(X)
        X_2 = self.three_by_three_block(X)
        X_3 = self.five_by_five_block(X)
        X_4 = self.pool_block(X)

        return torch.cat([X_1, X_2, X_3, X_4], dim=1)




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

class GoogLeNet(nn.Module):
    def __init__(self, num_of_classes = 200, input_size = 224, num_of_channels = 3):
        super(GoogLeNet, self).__init__()
        self.input_size = input_size

        self.downsampling = nn.Sequential(
            ConvNormRelu(num_of_channels, out_channels=64, kernel_size=7, stride = 2, padding = 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ConvNormRelu(64, 64, kernel_size=1, stride = 1, padding = 0),
            ConvNormRelu(64, 192, kernel_size=3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        in_channels= [192,256,"M",480,512,512,512,528,"M",832,832]

        dimensions_1x1 = [64,128,"M",192,160,128,112,256,"M",256,384]
        dimensions_3x3_reduce = [96,128, "M",96,112,128,144,160,"M",160,192]
        dimensions_3x3 = [128,192,"M",208,224,256,288,320,"M",320,384]
        dimensions_5x5_reduce = [16,32,"M",16,24,24,32,32,"M",32,48]
        dimensions_5x5 = [32,96,"M",48,64,64,64,128,"M",128,128]
        dimensions_pool = [32,64,"M",64,64,64,64,128,"M",128,128]

        self.stacked_inceptions = nn.ModuleList()

        for i in range(len(dimensions_1x1)):
            if dimensions_1x1[i] == "M":
                self.stacked_inceptions.append(nn.MaxPool2d(3,stride=2, padding=1))
            else:
                self.stacked_inceptions.append(
                    InceptionModule(in_channels[i], dimensions_1x1[i], dimensions_3x3_reduce[i],
                    dimensions_3x3[i], dimensions_5x5_reduce[i],dimensions_5x5[i],
                    dimensions_pool[i])
                )

        self.output_spatial_size = self.calculate_spatial_dim()
        self.output_channel_size = 1024
        self.prediction_head = nn.Sequential(
            nn.AvgPool2d(self.output_spatial_size, stride=1),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(self.output_channel_size, num_of_classes)
        )

    def forward(self, X):
        X = self.downsampling(X)
        #print(X.shape)
        for inception in self.stacked_inceptions:
            X = inception(X)
            #print(X.shape)

        return self.prediction_head(X)
        

        

    def calculate_spatial_dim(self):
        batch = torch.randn(1, 3, self.input_size, self.input_size)
        batch = self.downsampling(batch)
        spatial_size = batch.shape[2]
        #Then we will have 3 maxPoolings all of which decrease spatial dimension by 2 so we have //4
        return spatial_size // 4
        

if __name__ == '__main__':
    googlenet = GoogLeNet(200, 224, 3)
    input = torch.randn(1,3,224,224)
    print(googlenet.output_spatial_size)
    print(googlenet(input).shape)