import torchvision.models as models
import torch
from models.mlp_head import MLPHead
from models.NetworksBlocks import SSRN, SSNet_AEAE_UP, SSNet_AEAE_IN, SSNet_AEAE_SA, SSNet_AEAE_KSC
import torch.nn.functional as F

class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

class SSRN_(torch.nn.Module):
    def __init__(self, name=None):
        super(SSRN_, self).__init__()
        if name == 'UP':
            ssrn = SSRN(num_classes=9, k=49)
        elif name == 'IP':
            ssrn = SSRN(num_classes=16, k=97)
        elif name == 'SA':
            ssrn = SSRN(num_classes=16, k=99)
        elif name == 'KSC':
            ssrn = SSRN(num_classes=13, k=85)
        self.encoder = torch.nn.Sequential(*list(ssrn.children())[:-1])
        self.projection = MLPHead(in_channels=ssrn.fc.in_features,
                                  mlp_hidden_size=512,
                                  projection_size=128)

    def forward(self, x):
        h = self.encoder(x)
        h = F.avg_pool2d(h, h.size()[-1])
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)

class SSRN_sampling(torch.nn.Module):
    def __init__(self, name=None):
        super(SSRN_sampling, self).__init__()
        if name == 'UP':
            ssrn = SSRN(num_classes=9, k=23)
        elif name == 'IN':
            ssrn = SSRN(num_classes=16, k=47)
        elif name == 'SA':
            ssrn = SSRN(num_classes=16, k=99)
        elif name == 'KSC':
            ssrn = SSRN(num_classes=13, k=41)
        self.encoder = torch.nn.Sequential(*list(ssrn.children())[:-1])
        self.projection = MLPHead(in_channels=ssrn.fc.in_features,
                                  mlp_hidden_size=512,
                                  projection_size=128)

    def forward(self, x):
        h = self.encoder(x)
        h = F.avg_pool2d(h, h.size()[-1])
        h = h.view(h.shape[0], h.shape[1])
        return self.projection(h)

class SSTN_(torch.nn.Module):
    def __init__(self, name=None):
        super(SSTN_, self).__init__()
        if name == 'UP':
            ssrn = SSNet_AEAE_UP()
        elif name == 'IP':
            ssrn = SSNet_AEAE_IN()
        elif name=='SA':
            ssrn = SSNet_AEAE_SA()
        elif name=='KSC':
            ssrn = SSNet_AEAE_KSC()
        self.encoder = torch.nn.Sequential(*list(ssrn.children())[:-1])
        self.projection = MLPHead(in_channels=ssrn.fc.in_features,
                                  mlp_hidden_size=512,
                                  projection_size=128)

    def forward(self, x):
        h = self.encoder(x)
        h = F.avg_pool2d(h, h.size()[-1])
        h = h.squeeze()
        return self.projection(h)

class SSRN_FN(torch.nn.Module):
    def __init__(self, name=None):
        super(SSRN_FN, self).__init__()
        if name == 'UP':
            ssrn = SSRN(num_classes=9, k=49)
            self.n_cls = 9
        elif name == 'IP':
            ssrn = SSRN(num_classes=16, k=97)
            self.n_cls = 16
        elif name == 'KSC_IP':
            ssrn = SSRN(num_classes=16, k=85)
            self.n_cls = 16
        elif name == 'KSC_UP':
            ssrn = SSRN(num_classes=9, k=85)
            self.n_cls = 9
        else:
            ssrn = SSRN(num_classes=16, k=99)
            self.n_cls = 16

        self.encoder = torch.nn.Sequential(*list(ssrn.children())[:-1])

        self.linear = torch.nn.Linear(ssrn.fc.in_features, self.n_cls )

    def forward(self, x):
        h = self.encoder(x)
        h = F.avg_pool2d(h, h.size()[-1])
        h = h.view(h.shape[0], h.shape[1])
        return self.linear(h)