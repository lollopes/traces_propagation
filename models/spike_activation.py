import torch
import torch.nn as nn
import torch.autograd as autograd

class SpikingActivationFunc(autograd.Function):
    scale = 1.0     # Adjusted scale factor for the surrogate gradient
    surrogate_type = "1"

    @staticmethod
    def forward(ctx, membrane_potential, threshold):
        spike = (membrane_potential >= threshold).float()
        ctx.save_for_backward(membrane_potential, threshold)
        return spike

    @staticmethod
    def backward(ctx, grad_spike):
        membrane_potential, threshold = ctx.saved_tensors
        grad_input = grad_spike.clone()

        if SpikingActivationFunc.surrogate_type == "1":
            surrogate_gradient = grad_input * SpikingActivationFunc.scale / (1 + (torch.pi * (membrane_potential - threshold)).pow(2))
        elif SpikingActivationFunc.surrogate_type == "2":
            surrogate_gradient = grad_input / (SpikingActivationFunc.scale * torch.abs(membrane_potential - threshold)  + 1.0) ** 2
        elif SpikingActivationFunc.surrogate_type == "3":
            delta =  SpikingActivationFunc.scale * torch.clamp(1.0 - torch.abs(membrane_potential - threshold), min=0.0)
            surrogate_gradient = grad_input * delta
      
        return surrogate_gradient, None

class SpikeActivation(nn.Module):
    def __init__(self, threshold=1.0, surrogate_type="1", surrogate_scale = 1.0, size = 1):
        super(SpikeActivation, self).__init__()
        self.threshold = torch.tensor(threshold)
        self.surrogate_type = surrogate_type
        SpikingActivationFunc.surrogate_type = surrogate_type
        SpikingActivationFunc.scale = surrogate_scale
        
    def forward(self, membrane_potential):
        spike = SpikingActivationFunc.apply(membrane_potential, self.threshold)
        return spike
