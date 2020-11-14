import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):
    def __init__(self, model_name, pretrained, num_classes=6):
        '''
        torchvision 
        [Implemented]
        resnet18, resnet34, resnet50, resnet101, resnet152
        resnext50_32x4d, resnext101_32x8d
        wide_resnet50_2, wide_resnet101_2
        densenet121, densenet169, densenet161, densenet201
        inception_v3, googlenet, 

        [NotImplementedYet]
        alexnet, vgg16, vgg16_bn, vgg19, vgg19_bn
        squeezenet1_0, squeezenet1_1
        shufflenet_v2_x0_5 shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
        mobilenet_v2, 

        '''
        super().__init__()
        #out name
        #fc: resnet, densenet, inception, googlenet, shufflenet, resnext50_32x4d, wide_resnet50_2
        #classifier: alexnet, vgg, squeezenet, mobilenet, mnasnet

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        last_fc = 'res' in self.model_name or \
                  'densenet' in self.model_name or \
                  self.model_name is 'inception_v3' or\
                  self.model_name is 'googlenet'
        exec('model_function = models.{}'.format(self.model_name))

        self.model = model_function(pretrained=self.pretrained)
        if last_fc:
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features: self.num_classes, bias=self.model.fc.bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)