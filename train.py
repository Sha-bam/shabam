import click
import os
import sys
import requests
import zipfile
import torch

class ModelArgs(dict):
    def __init__(self, *args, **kwargs):
      super(ModelArgs, self).__init__(*args, **kwargs)
      self.__dict__ = self

@click.command()
@click.option('--epochs', default=100, 
	help='Number of training epochs.')
@click.option('--lr', default=1e-3, help='Initial learning rate.')
@click.option('--cuda/--no-cuda', default=True,
	help='Train on GPU if available.')
@click.option('--beta', nargs=2, type=(float, float), 
	help='Beta range for Adam optimizer.')

def main(epochs, lr, cuda, beta):
    if cuda:
        if not torch.cuda.is_available():
            print("Cuda not available")
            return
        
        device = torch.device('cuda')

    args = ModelArgs({
        'epochs': epochs,
        'lr': lr,
        'beta': beta,
        'device': device
    })
    print(epochs, lr, beta, device, download_data)
    return args


if __name__ == '__main__':
	main()