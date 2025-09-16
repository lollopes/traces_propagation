import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neuron_layers import LILayer, LIFLayer


class BP_MLP(nn.Module):
    def __init__(self, hidden_layers, input_size, output_size, batch_size, snn_params):
        super(BP_MLP, self).__init__()
        self._initialize_params(hidden_layers, input_size, output_size, batch_size, snn_params)
        self._build_network(hidden_layers, input_size, output_size, batch_size, snn_params)


        self.cumulative_spikes = [0 for _ in range(len(self.network)-1)]
        self.total_spike_neurons = [0 for _ in range(len(self.network)-1)]

        self.cumulative_spikes_target = [0 for _ in range(len(self.network)-1)]
        self.total_spike_neurons_target = [0 for _ in range(len(self.network)-1)]

    def _initialize_params(self, hidden_layers, input_size, output_size, batch_size, snn_params):
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.leak_t = snn_params["l_leak_t"]
        self.surrogate_type = snn_params["surrogate_type"]
        self.surrogate_scale = snn_params["surrogate_scale"]

    def _build_network(self, hidden_layers, input_size, output_size, batch_size, snn_params):
        self.network = nn.ModuleList()
        # First layer
        self.network.append(self._create_lif_layer(input_size, hidden_layers[0], batch_size, snn_params))
        
        # Hidden layers 
        for i in range(1, len(hidden_layers)):
            self.network.append(self._create_lif_layer(hidden_layers[i-1], hidden_layers[i], batch_size, snn_params))
        
        # Last layer (integrator)
        self.network.append(LILayer(hidden_layers[-1], output_size, batch_size, snn_params["l_out_leak_m"], snn_params["norm_type"]))
            

    def _create_lif_layer(self, input_size, output_size, batch_size, snn_params):
        return LIFLayer(
            input_size=input_size,
            output_size=output_size,
            batch_size=batch_size,
            vth=snn_params["l_vth"],
            leak_m=snn_params["l_leak_m"],
            leak_t=snn_params["l_leak_t"],
            recurrent=snn_params["l_rec"],
            reset_type=snn_params["l_rst_type"],
            surrogate_type=self.surrogate_type,
            surrogate_scale=self.surrogate_scale,
            norm = snn_params["norm_type"],
        )

    def forward(self, x, target):
        x = x.view(x.shape[0], -1)

        spike, mem, trace = self.network[0](x)
        self._spike_summer(0, spike, spike)

        for i in range(1, len(self.network)-1):
            spike, mem, trace = self.network[i](spike)
            self._spike_summer(i, spike, spike)

        mem_out = self.network[-1](spike)

        return mem_out, mem_out

    def inference(self, x):
        x = x.view(x.shape[0], -1)

        spike, mem, trace = self.network[0](x)

        for i in range(1, len(self.network)-1):
            spike, mem, trace = self.network[i](spike)

        mem_out = self.network[-1](spike)

        return mem_out

    def reset_potential(self):
        for layer in self.network:
            layer.reset()

    def _spike_summer(self, layer_idx, spike_in, spike_tgt):
        self.cumulative_spikes[layer_idx]          += spike_in.sum().item()
        self.total_spike_neurons[layer_idx]        += spike_in.numel()

    def _print_spike_rates(self):
        rates = (torch.tensor(self.cumulative_spikes, dtype=torch.float32) /
                torch.tensor(self.total_spike_neurons, dtype=torch.float32)
                ) * 100
        print("layer-wise spike rates (%) :", rates.round(decimals=2).tolist())

    def _reset_spike_counters(self):
        for i in range(len(self.cumulative_spikes)):
            self.cumulative_spikes[i]   = 0
            self.total_spike_neurons[i] = 0
            
    def get_layerwise_params(self):
        layerwise_params = []
        for i, layer in enumerate(self.network):
            params = list(layer.parameters())
            if params:  # Only include layers with trainable parameters
                layerwise_params.append((f"layer_{i}", params))

        return layerwise_params
    