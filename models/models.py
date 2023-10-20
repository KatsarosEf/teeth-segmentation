import torch.nn as nn
import segmentation_models_pytorch as smp

class UnetPlusPlus(nn.Module):
    def __init__(self, tasks, classes):
        super(UnetPlusPlus, self).__init__()
        self.tasks = tasks
        self.model = smp.UnetPlusPlus(classes=classes, encoder_name='resnet50', in_channels=3, encoder_weights='imagenet')

        for sm in self.model.modules():
            if isinstance(sm, nn.Conv2d):
                # only conv2d will be initialized in this way
                nn.init.normal_(sm.weight.data, 0.0, 0.02)
            elif isinstance(sm, nn.BatchNorm2d):
                nn.init.constant_(sm.weight, 1)
                nn.init.constant_(sm.bias, 0)


    def forward(self, x):
        return (self.model(x),)



class DeepLabV3Plus(nn.Module):
    def __init__(self, tasks, classes):

        super(DeepLabV3Plus, self).__init__()
        self.tasks = tasks
        self.model = smp.DeepLabV3Plus(classes=classes, encoder_name='resnet50', in_channels=3, encoder_weights='imagenet')

        for sm in self.model.modules():
            if isinstance(sm, nn.Conv2d):
                # only conv2d will be initialized in this way
                nn.init.normal_(sm.weight.data, 0.0, 0.02)
            elif isinstance(sm, nn.BatchNorm2d):
                nn.init.constant_(sm.weight, 1)
                nn.init.constant_(sm.bias, 0)


    def forward(self, x):
        return self.model(x)