import torch.nn as nn

from labml import experiment
from labml.configs import calculate
from labml_helpers.module import Module
from labml_nn.gan.original.experiment import Configs, Discriminator

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(Module):
    def __init__(self):
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 3, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1024, 512, 3, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(_weights_init)
        
    def forward(self, x):
        x = self.layers(x)
        return x.view(x.shape[0], -1)
    
calculate(Configs.generator, 'cnn', lambda c: Generator().to(c.device))
calculate(Configs.discriminator, 'cnn', lambda c: Discriminator().to(c.device))

def main():
    conf = Configs()
    experiment.create(name='mnist_dcgan')
    experiment.configs(conf,
                       {'discriminator' : 'cnn',
                        'generator' : 'cnn',
                        'label_smoothing' : 0.01})
    with experiment.start():
        conf.run()
        
if __name__ == '__main__':
    main()