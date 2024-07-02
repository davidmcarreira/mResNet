import torch
from torch import nn
from networks.Bottleneck import Bottleneck

class ResNet(nn.Module):
    def __init__(self, variant, in_channels, num_classes):
        super(ResNet, self).__init__()
        #self.variant_list = variant
        self.channels_blocks = variant[0]
        self.blocks_rep = variant[1]
        self.expansion = variant[2]
        self.is_Bottleneck = variant[3] 
        
        self.input_channels = in_channels
        
        # Following the scheme in ResNet's paper Table 1
        self.conv1 = nn.Conv2d(in_channels = in_channels, 
                               out_channels = 64, 
                               kernel_size = 7, 
                               stride = 2, 
                               padding = 3, 
                               bias = False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.max_pool = nn.MaxPool2d(kernel_size = 3,
                                  stride = 2,
                                  padding = 1)
        
        
        self.block1 = self.block_gen(64, self.channels_blocks[0], self.blocks_rep[0], self.expansion, self.is_Bottleneck, stride = 1)
        self.block2 = self.block_gen(self.channels_blocks[0]*self.expansion, self.channels_blocks[1], self.blocks_rep[1], self.expansion, self.is_Bottleneck, stride = 2)
        self.block3 = self.block_gen(self.channels_blocks[1]*self.expansion, self.channels_blocks[2], self.blocks_rep[2], self.expansion, self.is_Bottleneck, stride = 2)
        self.block4 = self.block_gen(self.channels_blocks[2]*self.expansion, self.channels_blocks[3], self.blocks_rep[3], self.expansion, self.is_Bottleneck, stride = 2)
        
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.channels_blocks[3]*self.expansion, num_classes)
        
    def block_gen(self, in_channels, inter_channels, channels_blocks, expansion, is_Bottleneck, stride):
        layers = []
        layers.append(Bottleneck(in_channels, inter_channels, expansion, is_Bottleneck, stride))
        for i in range(1, channels_blocks):
            layers.append(Bottleneck(inter_channels*expansion, inter_channels, expansion, is_Bottleneck, stride = 1))
        #print(len(layers))
        #print(layers)
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.block1(x) # Bottleneck layers
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.average_pool(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)
        
        return x