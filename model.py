import torch.nn as nn
from PIL import Image
import torch


class Encoder(nn.Module):
    def __init__(self, vgg_path):
        super(Encoder, self).__init__()

        self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1,1)),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(3, 64, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),

        nn.Conv2d(64,64, (3,3)),
        nn.ReLU(),
        nn.MaxPool2d((2,2),(2,2),(0,0), ceil_mode=True),
        nn.ReflectionPad2d((1,1,1,1)),

        nn.Conv2d(64, 128, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(128,128, (3,3)),
        nn.ReLU(),
        nn.MaxPool2d((2,2), (2,2),(0,0), ceil_mode=True),
        nn.ReflectionPad2d((1,1,1,1)),

        nn.Conv2d(128, 256, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(256, 256, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(256, 256, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(256, 256, (3,3)),
        nn.ReLU(),
        nn.MaxPool2d((2,2), (2,2), (0,0), ceil_mode=True),
        nn.ReflectionPad2d((1,1,1,1)),

        nn.Conv2d(256, 512, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(512, 512, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(512, 512, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(512, 512, (3,3)),
        nn.ReLU(),
        nn.MaxPool2d((2,2), (2,2), (0,0), ceil_mode=True),
        nn.ReflectionPad2d((1,1,1,1)),

        nn.Conv2d(512, 512, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(512, 512, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(512, 512, (3,3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1,1,1,1)),
        nn.Conv2d(512, 512, (3,3)),
        nn.ReLU()
        )

        self.vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        enc_layers = list(self.vgg.children())
        self.enc1 = nn.Sequential(*enc_layers[:4])
        self.enc2 = nn.Sequential(*enc_layers[4:11])
        self.enc3 = nn.Sequential(*enc_layers[11:18])
        self.enc4 = nn.Sequential(*enc_layers[18:31])

        for name in ['enc1', 'enc2', 'enc3', 'enc4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def forward(self, input, is_test = False):
        h1 = self.enc1(input)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)

        if is_test:
            return h4

        return h1, h2, h3, h4
    



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.net = nn.Sequential(
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(512, 256, (3,3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, input):
        return self.net(input)