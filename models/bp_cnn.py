import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neuron_layers import LILayer, LIFLayerCNN


VGG_CONFIGS = {
    6: [  # 5 conv + 1 FC
        (64, 1, "max"),
        (128, 2, "max"),
        (256, 1, "max"),
        (512, 2, "max"),
        (512, 2, "aavg"),
    ],
    9: [  # 8 conv + 1 FC
        (64, 1, "max"),
        (128, 2, "max"),
        (256, 1, "max"),
        (256, 2, "max"),
        (512, 1, "max"),
        (512, 2, "max"),
        (512, 1, "max"),
        (512, 2, "aavg"),
    ],
    12: [  # 11 conv + 1 FC
        (64, 1, "max"),
        (64, 2, "max"),
        (128, 1, "max"),
        (128, 2, "max"),
        (256, 1, "max"),
        (256, 1, "max"),
        (256, 2, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 2, "aavg"),
    ],
    15: [  # 14 conv + 1 FC
        (64, 1, "max"),
        (64, 1, "max"),
        (128, 2, "max"),
        (128, 1, "max"),
        (256, 1, "max"),
        (256, 1, "max"),
        (256, 2, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 2, "max"),
        (512, 1, "max"),
        (512, 1, "max"),
        (512, 2, "max"),
        (512, 2, "aavg"),
    ]
}

class BP_CNN(nn.Module):
    def __init__(self, input_h, input_w, input_channels, output_size, batch_size, snn_params):
        super(BP_CNN, self).__init__()
        self._initialize_params(input_h, input_w, input_channels, output_size, snn_params)
        self._build_network(input_h, input_w, input_channels, batch_size)
        self._init_spike_counters()

    def _initialize_params(self, input_h, input_w, input_channels, output_size, snn_params):

        self.input_h = input_h
        self.input_w = input_w
        self.input_channels = input_channels
        self.output_size = output_size

        self.leak_t = snn_params["l_leak_t"]
        self.leak_m = snn_params["l_leak_m"]
        self.vth = snn_params["l_vth"]
        self.reset_type = snn_params["l_rst_type"]

        self.leak_m_out = snn_params["l_out_leak_m"]

        self.surrogate_type = snn_params["surrogate_type"]
        self.surrogate_scale = snn_params["surrogate_scale"]

        self.vgg_variant = snn_params["vgg_variant"]
        self.norm_type = snn_params["norm_type"]

    def _build_network(self, input_h, input_w, input_channels, batch_size):
        self.network = nn.ModuleList()
        curr_h, curr_w, curr_channels = input_h, input_w, input_channels

        def add_block(out_channels, kernel_size, padding, pool_size, double_batch=False, pool_type="max"):
            nonlocal curr_h, curr_w, curr_channels

            next_h = self._conv_output_size(curr_h, kernel_size, padding)
            next_w = self._conv_output_size(curr_w, kernel_size, padding)

            if pool_type == "aavg":
                next_h = self._pool_output_size(next_h, pool_size, adaptive=True)
                next_w = self._pool_output_size(next_w, pool_size, adaptive=True)
                pool_kernel_h = pool_kernel_w = pool_size
            elif pool_size > 1:
                pool_kernel_h = pool_kernel_w = pool_size
                next_h = self._pool_output_size(next_h, pool_kernel_h)
                next_w = self._pool_output_size(next_w, pool_kernel_w)
            else:
                pool_kernel_h = pool_kernel_w = 1  # no pooling

            this_batch = batch_size * 2 if double_batch else batch_size

            self.network.append(LIFLayerCNN(
                in_w=curr_w,
                in_h=curr_h,
                in_channels=curr_channels,
                out_channels=out_channels,
                batch_size=this_batch,
                kernel_size=kernel_size,
                padding=padding,
                pool_size=(pool_kernel_h, pool_kernel_w),
                pool_type=pool_type,
                vth=self.vth,
                leak_m=self.leak_m,
                leak_t=self.leak_t,
                reset_type=self.reset_type,
                surrogate_type=self.surrogate_type,
                surrogate_scale=self.surrogate_scale,
                norm=self.norm_type,
            ))

            curr_h, curr_w, curr_channels = next_h, next_w, out_channels

        # Build convolutional backbone
        vgg_cfg = VGG_CONFIGS.get(self.vgg_variant)
        assert vgg_cfg is not None, f"Unsupported VGG variant: {self.vgg_variant}"

        for out_ch, pool_sz, pool_type in vgg_cfg:
            add_block(out_ch, 3, 1, pool_sz, pool_type=pool_type)

        # Final FC layer
        self.network.append(LILayer(
            input_size=curr_h * curr_w * curr_channels,
            output_size=self.output_size,
            batch_size=batch_size,
            leak=self.leak_m_out,
            norm=self.norm_type
        ))

    def _forward_pass(self, x):
        # first spiking layer
        spike, _, _   = self.network[0](x)
        self._spike_summer(0, spike)
        out = spike

        # remaining spiking layers – skip the final FC
        for idx, layer in enumerate(self.network[1:-1], start=1):
            spike, _, _ = layer(out)
            self._spike_summer(idx, spike)
            out = spike

        # final (non-spiking) fully-connected layer
        mem = self.network[-1](out.flatten(1))   # only membrane values returned
        return mem


    def forward(self, x, _):
        return self._forward_pass(x), self._forward_pass(x)

    def inference(self, x):
        return self._forward_pass(x)

    def get_layerwise_params(self):
        layerwise_params = []
        for i, layer in enumerate(self.network):
            params = list(layer.parameters())
            if params:  # Only include layers with trainable parameters
                layerwise_params.append((f"layer_{i}", params))

        return layerwise_params
    
    def reset_potential(self):
        for layer in self.network:
            layer.reset()

        self._reset_spike_counters()     

    def _conv_output_size(self, size, kernel_size, padding, stride=1):
        return (size + 2 * padding - kernel_size) // stride + 1

    def _pool_output_size(self, size, pool_size, stride=None, adaptive=False):
        if adaptive:
            return pool_size
        else:
            if stride is None:
                stride = pool_size
            return (size - pool_size) // stride + 1

    def _init_spike_counters(self):
        L = len(self.network)
        self.cumulative_spikes      = [0 for _ in range(L-1)]
        self.total_spike_neurons    = [0 for _ in range(L-1)]

    def _spike_summer(self, idx: int, spike: torch.Tensor):
        self.cumulative_spikes[idx]   += spike.sum().item()
        self.total_spike_neurons[idx] += spike.numel()

    def _print_spike_rates(self):
        rates = (torch.tensor(self.cumulative_spikes, dtype=torch.float32) /
                torch.tensor(self.total_spike_neurons, dtype=torch.float32)
                ) * 100
        print("layer-wise spike rates (%) :", rates.round(decimals=2).tolist())

    def _reset_spike_counters(self):
        for i in range(len(self.cumulative_spikes)):
            self.cumulative_spikes[i]   = 0
            self.total_spike_neurons[i] = 0
