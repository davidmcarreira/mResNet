import torch

# all nn libraries nn.layer, convs and loss functions
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, inter_channels, expansion, is_Bottleneck, stride):
        super(Bottleneck, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.expansion = expansion
        self.is_Bottleneck = is_Bottleneck # 1x1 -> 3x3 -> 1x1
        
        if self.in_channels == inter_channels*expansion: # Verifies if the input channels size is equal to the output size
            self.identity = True # Perform identity mapping
        else:
            self.identity = False # Project the input feature maps to the required dimensions 
            
            layers = []
            layers.append(nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.inter_channels*self.expansion, 
                                   kernel_size=1, stride=stride, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(self.inter_channels*self.expansion))
            self.projection = nn.Sequential(*layers)
            
        self.relu = nn.ReLU()
        
        if self.is_Bottleneck: # For Resnet version with more than 50 layers
            self.conv1_x = nn.Conv2d(in_channels = self.in_channels,
                                  out_channels = self.inter_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  padding = 0,
                                  bias = False)
            self.batch_norm1_x = nn.BatchNorm2d(self.inter_channels)
            
            self.conv2_x = nn.Conv2d(in_channels = self.inter_channels,
                                    out_channels = self.inter_channels,
                                    kernel_size = 3,
                                    stride = stride,
                                    padding = 1,
                                    bias = False)
            self.batch_norm2_x = nn.BatchNorm2d(self.inter_channels)
            
            self.conv3_x = nn.Conv2d(in_channels = self.inter_channels,
                                    out_channels = self.inter_channels*self.expansion,
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = 0,
                                    bias = False)
            self.batch_norm3_x = nn.BatchNorm2d(self.inter_channels*self.expansion)
        else:
            self.conv1_x = nn.Conv2d(in_channels = self.in_channels,
                                  out_channels = self.inter_channels,
                                  kernel_size = 3,
                                  stride = stride,
                                  padding = 1,
                                  bias = False)
            self.batch_norm1_x = nn.BatchNorm2d(self.inter_channels)
            
            self.conv2_x = nn.Conv2d(in_channels = self.inter_channels,
                                  out_channels = self.inter_channels,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = 1,
                                  bias = False)
            self.batch_norm2_x = nn.BatchNorm2d(self.inter_channels)
            
    def forward(self, x):
        input_x = x
        
        if self.is_Bottleneck:
            x = self.relu(self.batch_norm1_x(self.conv1_x(x))) # conv 1x1 -> Batch Norm -> ReLU
            x = self.relu(self.batch_norm2_x(self.conv2_x(x))) # conv 3x3 -> Batch Norm -> ReLU
            x = self.batch_norm3_x(self.conv3_x(x)) # conv 1x1 -> Batch Norm
            
        else:
            x = self.relu(self.batch_norm1_x(self.conv1_x(x))) # conv 3x3 -> Batch Norm -> ReLU
            x = self.batch_norm2_x(self.conv2_x(x))
            
        if self.identity:
            x += input_x
        else:
            x += self.projection(input_x)
            
        x = self.relu(x) # Last ReLU
        
        return x