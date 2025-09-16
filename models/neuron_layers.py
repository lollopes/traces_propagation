import torch
import torch.nn as nn
from models.spike_activation import SpikeActivation
import torch.nn.functional as F


class Norm:
    def _setup_norm(self, weight_tensor, norm):

        if norm not in ("weight", "layer", "none"):
            raise ValueError("norm must be 'weight' or 'layer' or 'none")

        self.norm_type = norm

        if self.norm_type == "weight":
            nn.init.kaiming_normal_(weight_tensor)

    def _resize_state(self, B, device):   # ← self, B, device
        """Resize internal state buffers to the new batch size B."""
        for name in ('membrane_potential', 'previous_spike', 'trace'):
            if hasattr(self, name):
                buf = getattr(self, name)
                if buf.size(0) != B:
                    new_shape = (B, *buf.shape[1:])
                    new_buf = torch.zeros(new_shape,
                                         dtype=buf.dtype,
                                         device=device)
                    self.register_buffer(name, new_buf)

        if hasattr(self, 'batch_size'):
            self.batch_size = B
            
    @staticmethod        
    def _ws(weight, eps = 1e-5):

        if weight.dim() == 4:
            dims, fan_in = (1, 2, 3), weight[0].numel()     # C·kH·kW
        elif weight.dim() == 2:
            dims, fan_in = (1,),     weight.size(1)         # in_features
        else:                                              # odd shape
            return weight

        mean = weight.mean(dim=dims, keepdim=True)
        var  = weight.var (dim=dims, keepdim=True, unbiased=False)
        return (weight - mean) / torch.sqrt(var * fan_in + eps)

    @staticmethod
    def _tensor_norm(x, eps = 1e-5):
        if x.dim() == 2:
            dims = (1,)
        elif x.dim() == 4:
            dims = (1, 2, 3)
        else:
            raise ValueError("tensor_norm supports 2-D or 4-D tensors")

        mean = x.mean(dim=dims, keepdim=True)
        std  = x.std (dim=dims, keepdim=True)
        return (x - mean) / (std + eps)

    def _maybe_ws(self, w):
        return 1.8 * self._ws(w) if self.norm_type == "weight" else w

    def _maybe_ln(self, x):
        return self._tensor_norm(x) if self.norm_type == "layer" else x    

class LILayer(Norm, nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 batch_size,
                 leak=0.9,
                 norm="layer"
                 ):
        
        super(LILayer, self).__init__()

        self.register_buffer('membrane_potential', torch.zeros(batch_size, output_size))
        self.leak = leak
        self.fc = nn.Linear(input_size, output_size, bias=False) 
        self._setup_norm(self.fc.weight, norm=norm)
        self.recurrent =  False

    def forward(self, x):
        B = x.size(0)
        self._resize_state(B, x.device)
        if self.norm_type == "none":
            input_current = self.fc(x)                      
        else:
            w = self._maybe_ws(self.fc.weight)
            input_current = F.linear(x, w, bias=None)
            input_current = self._maybe_ln(input_current)    

        self.membrane_potential = self.leak * self.membrane_potential + input_current

        return self.membrane_potential
    
    def reset(self):
        self.membrane_potential = torch.zeros_like(
            self.membrane_potential,
            device=self.membrane_potential.device,
        )

class LIFLayer(Norm, nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 batch_size,
                 vth = 1.0, 
                 leak_m=0.9,
                 leak_t=0.9,
                 recurrent = False,
                 reset_type = "soft",
                 surrogate_type = "2",
                 surrogate_scale = 1.0,
                 norm="weight",
                 ):
        
        super(LIFLayer, self).__init__()

        self.register_buffer('membrane_potential', torch.zeros(batch_size, output_size))
        self.register_buffer('previous_spike', torch.zeros(batch_size, output_size))
        self.register_buffer('trace', torch.zeros(batch_size, output_size))

        self.reset_type = reset_type
        self.vth = vth
        self.surrogate_type = surrogate_type
        self.activation = SpikeActivation(self.vth, self.surrogate_type, surrogate_scale, output_size)
        self.recurrent = recurrent

        self.leak_m = leak_m
        self.leak_t = leak_t
        self.fc = nn.Linear(input_size, output_size, bias=False) 
        self._setup_norm(self.fc.weight, norm=norm)

        self.fc = nn.Linear(input_size, output_size, bias=False)
        self._setup_norm(self.fc.weight, norm=norm)

        if self.recurrent:
            self.rc = nn.Linear(output_size, output_size, bias = False)
            

    def forward(self, x):
        B = x.size(0)
        self._resize_state(B, x.device)
        if self.norm_type == "none":
            input_current = self.fc(x)
        else:
            w = self._maybe_ws(self.fc.weight)
            input_current = F.linear(x, w, bias=None)
            input_current = self._maybe_ln(input_current)

        if self.recurrent: 
            input_current += self.rc(self.previous_spike)
        
        # 1) Leak + drive
        membrane_potential_pre = self.leak_m * self.membrane_potential + input_current
        # 2) Fire
        spike = self.activation(membrane_potential_pre)
        tmp = spike
            
        # 3) Reset (soft vs hard)
        if self.reset_type == "soft":
            membrane_potential_new = membrane_potential_pre - spike * self.vth
        else:  # hard reset
            membrane_potential_new = membrane_potential_pre * (1 - spike)
        # 4) Update membrane potential for next time step 
        self.membrane_potential = membrane_potential_new
        # 5) Save spike for recurent connection to be integrated in the next time step
        self.previous_spike = spike.clone()
        # 6) Compute trace
        self.trace = self.trace * self.leak_t + tmp

        return spike, membrane_potential_new, self.trace
    
    def reset(self):
        self.membrane_potential = torch.zeros_like(
            self.membrane_potential,
            device=self.membrane_potential.device,
        )
        self.trace = torch.zeros_like(
            self.trace,
            device=self.trace.device,
        )
        self.previous_spike = torch.zeros_like(
            self.membrane_potential,
            device=self.trace.device,
        )

class LIFLayerCNN_PROP(Norm, nn.Module):
    def __init__(self, 
                 in_h,
                 in_w,
                 in_channels,
                 out_channels,
                 target_dim,
                 batch_size,
                 kernel_size,
                 padding,
                 pool_size,
                 pool_type,
                 vth = 1.0, 
                 leak_m=0.9,
                 leak_t=0.9,
                 reset_type="Soft",
                 surrogate_type = "2",
                 surrogate_scale = 1.0,
                 norm="weight"
                 ):
        
        super(LIFLayerCNN_PROP, self).__init__()

        if pool_type == "max":
            if pool_size[0] == 1: 
                self.pool_layer = nn.Identity()
            else:
                self.pool_layer = nn.MaxPool2d(pool_size)
        elif pool_type == "avg":
            if pool_size[0] == 1: 
                self.pool_layer = nn.Identity()
            else:
                self.pool_layer = nn.AvgPool2d(pool_size)
        elif pool_type == "aavg":
                self.pool_layer = nn.AdaptiveAvgPool2d(output_size=pool_size)


        self.out_channels = out_channels
        self.batch_size = batch_size

        self.out_h_pool = pool_size[0] if pool_type=="aavg" else ((in_h + 2*padding - kernel_size + 1 - pool_size[0]) // pool_size[0] + 1)
        self.out_w_pool = pool_size[1] if pool_type=="aavg" else ((in_w + 2*padding - kernel_size + 1 - pool_size[1]) // pool_size[1] + 1)

        self.output_size =  out_channels * ((in_h + 2 * padding - kernel_size) + 1) * ((in_w + 2 * padding - kernel_size) + 1)
        self.out_w = (in_w + 2 * padding - kernel_size) + 1
        self.out_h = (in_h + 2 * padding - kernel_size) + 1

        self.register_buffer('membrane_potential', torch.zeros(batch_size, self.output_size))
        self.register_buffer('previous_spike', torch.zeros(batch_size, self.output_size))
        self.register_buffer('trace', torch.zeros(batch_size, self.output_size))

        self.leak_m = leak_m
        self.leak_t = leak_t
        self.reset_type = reset_type
        self.vth = vth
        self.surrogate_type = surrogate_type
        self.activation = SpikeActivation(self.vth, self.surrogate_type, surrogate_scale, self.output_size)
        self.fc = nn.Linear(target_dim, self.output_size, bias=False) 
        self._setup_norm(self.fc.weight, norm=norm)


    def forward(self, x):
        B = x.size(0)
        self._resize_state(B, x.device)
        # Compute input current
        if self.norm_type == "none":
            current = self.fc(x)                     
            input_current = current.flatten(1)   
        else:
            w = self._maybe_ws(self.fc.weight)
            current = F.linear(x, w, bias=None)
            current = self._maybe_ln(current.reshape(B, self.out_channels, self.out_h, self.out_w)).flatten(1)                     
            input_current = current    

        # 1) Leak + drive
        membrane_potential_pre = self.leak_m * self.membrane_potential + input_current
        # 2) Fire
        spike = self.activation(membrane_potential_pre)
        # Reshape the spikes into an image and apply pooling         
        spike_img = spike.reshape(B, self.out_channels, self.out_h, self.out_w)
        spike_img = self.pool_layer(spike_img)
        tmp = spike_img.flatten(1)

        # Trace integration
        self.trace = self.trace * self.leak_t + tmp

        # 3) Reset (soft vs hard)
        if self.reset_type == "soft":
            membrane_potential_new = membrane_potential_pre - spike * self.vth
        else:  # hard reset
            membrane_potential_new = membrane_potential_pre * (1 - spike)
        # 4) Update membrane potential for next time step 
        self.membrane_potential = membrane_potential_new
        # 5) Save spike for recurent connection to be integrated in the next time step
        self.previous_spike = spike.clone()

        # Reshape trace potential and spikes as images 
        trace_img = self.trace.reshape(B, self.out_channels, self.out_h_pool, self.out_w_pool)
        membrane_img = self.membrane_potential.reshape(B, self.out_channels, self.out_h, self.out_w)

        return spike_img, membrane_img, trace_img
    
    def reset(self):
        self.membrane_potential = torch.zeros_like(
            self.membrane_potential,
            device=self.membrane_potential.device,
        )
        self.trace = torch.zeros_like(
            self.trace,
            device=self.trace.device,
        )
        self.previous_spike = torch.zeros_like(
            self.membrane_potential,
            device=self.trace.device,
        )

class LIFLayerCNN(Norm, nn.Module):
    def __init__(self, 
                 in_h,
                 in_w,
                 in_channels,
                 out_channels,
                 batch_size,
                 kernel_size,
                 padding,
                 pool_size,
                 pool_type,
                 vth = 1.0, 
                 leak_m=0.9,
                 leak_t=0.9,
                 reset_type="Soft",
                 surrogate_type = "2",
                 surrogate_scale = 1.0,
                 norm="weight"
                 ):
        
        super(LIFLayerCNN, self).__init__()

        if pool_type == "max":
            if pool_size[0] == 1: 
                self.pool_layer = nn.Identity()
            else:
                self.pool_layer = nn.MaxPool2d(pool_size)
        elif pool_type == "avg":
            if pool_size[0] == 1: 
                self.pool_layer = nn.Identity()
            else:
                self.pool_layer = nn.AvgPool2d(pool_size)
        elif pool_type == "aavg":
                self.pool_layer = nn.AdaptiveAvgPool2d(output_size=pool_size)
        
        # General parameters
        self.out_channels = out_channels
        self.batch_size = batch_size

        # Pooled output sizes
        self.out_h_pool = pool_size[0] if pool_type=="aavg" else ((in_h + 2*padding - kernel_size + 1 - pool_size[0]) // pool_size[0] + 1)
        self.out_w_pool = pool_size[1] if pool_type=="aavg" else ((in_w + 2*padding - kernel_size + 1 - pool_size[1]) // pool_size[1] + 1)
        self.output_size_flatten_pooled = self.out_channels * self.out_h_pool * self.out_w_pool
  
        # Regular output sizes
        self.out_w = (in_w + 2 * padding - kernel_size) + 1
        self.out_h = (in_h + 2 * padding - kernel_size) + 1
        self.output_size_flatten = self.out_channels * self.out_h * self.out_w

        # Neuron states 
        self.register_buffer('membrane_potential', torch.zeros(batch_size, self.output_size_flatten))
        self.register_buffer('previous_spike', torch.zeros(batch_size, self.output_size_flatten))
        self.register_buffer('trace', torch.zeros(batch_size, self.output_size_flatten_pooled))

        self.leak_m = leak_m
        self.leak_t = leak_t
        self.reset_type = reset_type
        self.vth = vth
        self.surrogate_type = surrogate_type
        self.activation = SpikeActivation(self.vth, self.surrogate_type, surrogate_scale, self.output_size_flatten)

        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self._setup_norm(self.fc.weight, norm=norm)

    def forward(self, x):
        B = x.size(0)
        self._resize_state(B, x.device)

        if self.norm_type == "none":
            cnn_out = self.fc(x)                    
            input_current = cnn_out.flatten(1)
        else:
            w = self._maybe_ws(self.fc.weight)
            cnn_out = F.conv2d(x, w,
                            bias=None,
                            stride=self.fc.stride,
                            padding=self.fc.padding,
                            dilation=self.fc.dilation,
                            groups=self.fc.groups)
            cnn_out = self._maybe_ln(cnn_out)      
            input_current = cnn_out.flatten(1)

        # 1) Leak + drive
        membrane_potential_pre = self.leak_m * self.membrane_potential + input_current
        # 2) Fire
        spike = self.activation(membrane_potential_pre)
        # Reshape the spikes into an image and apply pooling         
        spike_img = spike.reshape(B, self.out_channels, self.out_h, self.out_w)
        spike_img = self.pool_layer(spike_img)
        tmp = spike_img.flatten(1) 
        
        # Integrate trace
        self.trace = self.trace * self.leak_t + tmp

        # 3) Reset (soft vs hard)
        if self.reset_type == "soft":
            membrane_potential_new = membrane_potential_pre - spike * self.vth
        else:  # hard reset
            membrane_potential_new = membrane_potential_pre * (1 - spike)
        # 4) Update membrane potential for next time step 
        self.membrane_potential = membrane_potential_new
        # 5) Save spike for recurent connection to be integrated in the next time step
        self.previous_spike = spike.clone()

        # Reshape trace potential and spikes as images 
        trace_img = self.trace.reshape(B, self.out_channels, self.out_h_pool, self.out_w_pool)
        membrane_img = self.membrane_potential.reshape(B, self.out_channels, self.out_h, self.out_w)

        return spike_img, membrane_img, trace_img
    
    def reset(self):
        self.membrane_potential = torch.zeros_like(
            self.membrane_potential,
            device=self.membrane_potential.device,
        )
        self.trace = torch.zeros_like(
            self.trace,
            device=self.trace.device,
        )
        self.previous_spike = torch.zeros_like(
            self.membrane_potential,
            device=self.trace.device,
        )


        