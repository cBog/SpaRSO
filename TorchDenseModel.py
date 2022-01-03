from typing import List
import torch.nn as nn

LayerDef = namedtuple('LayerDef', [hidden_size, activation])

class DenseModel(nn.Module):

    def __init__(self, input_size, _layers: List[LayerDef]):
        self.layers = []

        previous_size = input_size
        for layerDef in _layers:
            layers.append(nn.Linear(previous_size, layerDef.hidden_size)
            previous_size = layerDef.hidden_size
            layers.append(layerDef.activation))


    def forward(self,x):

        for layer in self.layers:
            x = layer(x)

    