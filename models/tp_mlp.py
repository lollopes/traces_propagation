import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neuron_layers import LILayer, LIFLayer

class TP_MLP(nn.Module):
    def __init__(self, hidden_layers, input_size, output_size, batch_size, snn_params):
        super(TP_MLP, self).__init__()
        self._initialize_params(hidden_layers, output_size, snn_params)
        self._build_network(hidden_layers, input_size, output_size, batch_size, snn_params)
        self._initialize_buffers(batch_size, input_size, output_size)
        self._initialize_accumulators()


        self.cumulative_spikes = [0 for _ in range(len(self.network)-1)]
        self.total_spike_neurons = [0 for _ in range(len(self.network)-1)]

        self.cumulative_spikes_target = [0 for _ in range(len(self.network)-1)]
        self.total_spike_neurons_target = [0 for _ in range(len(self.network)-1)]

    def _initialize_params(self, hidden_layers, output_size, snn_params):
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.leak_t = snn_params["l_leak_t"]
        self.surrogate_type = snn_params["surrogate_type"]
        self.surrogate_scale = snn_params["surrogate_scale"]
        self.train_s = snn_params["train_s"]

    def _build_network(self, hidden_layers, input_size, output_size, batch_size, snn_params):
        self.network = nn.ModuleList()

        self.network.append(
            self._create_lif_layer(input_size, hidden_layers[0], batch_size, snn_params)
        )

        self.target_propagator = self._create_lif_layer(
            output_size, hidden_layers[0], batch_size, snn_params
        )

        for i in range(1, len(hidden_layers)):
            self.network.append(
                self._create_lif_layer(hidden_layers[i - 1], hidden_layers[i], batch_size * 2, snn_params)
            )

        self.network.append(
            LILayer(hidden_layers[-1], output_size, batch_size * 2, snn_params["l_out_leak_m"], snn_params["norm_type"])
        )

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

    def _initialize_buffers(self, batch_size, input_size, output_size):
        self.register_buffer('trace_in', torch.zeros(batch_size, input_size))
        self.register_buffer('trace_in_target', torch.zeros(batch_size, output_size))
        self.input = torch.Tensor()
        self.target = torch.Tensor()
        self.e = torch.Tensor()
        self.hs = []
        self.ts = []
        self.trace = []
        self.trace_target = []

    def _resize_state(self, B: int, device):

        for layer in self.network:
            if hasattr(layer, '_resize_state'):
                layer._resize_state(B, device)

        if self.trace_in.size(0) != B:
            self.trace_in = torch.zeros(
                B, self.trace_in.size(1), device=device, dtype=self.trace_in.dtype
            )
        if self.trace_in_target.size(0) != B:
            self.trace_in_target = torch.zeros(
                B, self.trace_in_target.size(1),
                device=device, dtype=self.trace_in_target.dtype
            )

    def _initialize_accumulators(self):
        self.accumulated_grads = {}

    def forward(self, x, target):
        self._reset_internal_states()
        B = x.size(0)

        x_flat = x.view(B, -1)
        self._update_traces(x_flat, target)
        spike_s, _, trace_s = self.network[0](x)             
        spike_t, __, trace_t = self.target_propagator(target) 


        self._store_internal_states(spike_s, spike_t, trace_s, trace_t)
        self._spike_summer(0, spike_s, spike_t)

        input_comb = torch.cat([spike_s.detach(), spike_t.detach()], dim=0)

        for idx in range(1, len(self.network) - 1):
            spike, _, trace = self.network[idx](input_comb)

            spike_s = spike[:B]
            spike_t = spike[B:]
            trace_s = trace[:B]
            trace_t = trace[B:]

            self._store_internal_states(spike_s, spike_t, trace_s, trace_t)
            self._spike_summer(idx, spike_s, spike_t)

            input_comb = spike.detach()

        mem_all = self.network[-1](input_comb)   
        mem_s, mem_t = mem_all[:B], mem_all[B:]

        self.e = nn.Softmax(dim=1)(mem_s) - target

        return mem_s, mem_t

    def inference(self, x):
        self._reset_internal_states()
        B = x.size(0)

        dummy_target = torch.zeros(B, self.output_size, device=x.device, dtype=x.dtype)
        x_flat = x.view(B, -1)
        self._update_traces(x_flat, dummy_target)

        spike_s, _, trace_s = self.network[0](x)
        self._store_internal_states(spike_s, None, trace_s, None)

        input_comb = torch.cat([spike_s.detach(), spike_s.detach()], dim=0)
        for idx in range(1, len(self.network) - 1):
            spike, _, trace = self.network[idx](input_comb)
            self._store_internal_states(spike[:B], None, trace[:B], None)
            input_comb = spike.detach()

        mem_all = self.network[-1](input_comb)
        return mem_all[:B]
        
    def _reset_internal_states(self):
        self.hs = []
        self.ts = []
        self.trace = []
        self.trace_target = []

    def _update_traces(self, x, target):
        B, Fin  = x.shape
        _, Ftar = target.shape
        if self.trace_in.size(0) != B:
            self.trace_in = torch.zeros(B, Fin,  device=x.device,     dtype=x.dtype)
        if self.trace_in_target.size(0) != B:
            self.trace_in_target = torch.zeros(B, Ftar, device=target.device,
                                            dtype=target.dtype)
        self.hs.append(x)
        self.ts.append(target)
        self.trace_in = self.leak_t * self.trace_in + x
        self.trace_in_target = self.leak_t * self.trace_in_target + target
        self.trace.append(self.trace_in)
        self.trace_target.append(self.trace_in_target)

    def _store_internal_states(self, spike, spike_targ, trace, trace_targ):
        self.hs.append(spike)
        self.ts.append(spike_targ)
        self.trace.append(trace)
        self.trace_target.append(trace_targ)

    def _spike_summer(self, layer_idx, spike_in, spike_tgt):
        self.cumulative_spikes[layer_idx]          += spike_in.sum().item()
        self.total_spike_neurons[layer_idx]        += spike_in.numel()
        self.cumulative_spikes_target[layer_idx]   += spike_tgt.sum().item()
        self.total_spike_neurons_target[layer_idx] += spike_tgt.numel()
    
    def update(self):
        bs = self.ts[0].shape[0]
        for i in range(len(self.network) - 1):

            loss, dist_trg, dist_in = self.loss(self.trace[i + 1], self.trace_target[i + 1], self.trace_target[i])

            if self.train_s  and i == 0:
                 grad_tp = torch.autograd.grad(loss, self.target_propagator.fc.weight, retain_graph=True)[0]  
                 self.target_propagator.fc.weight.grad = grad_tp

            grad_w = torch.autograd.grad(loss, self.network[i].fc.weight, retain_graph=True)[0]
            self.network[i].fc.weight.grad = grad_w

            if self.network[i].recurrent:
                grad_rec = torch.autograd.grad(loss, self.network[i].rc.weight, retain_graph=True)[0]
                self.network[i].rc.weight.grad = grad_rec

        grad_final = torch.einsum('bi,bj->ij', self.e, self.trace[-1]) / bs
        self.network[-1].fc.weight.grad = grad_final

    def get_layerwise_params(self):
        layerwise_params = []
        for i, layer in enumerate(self.network):
            params = list(layer.parameters())
            if params:  
                layerwise_params.append((f"layer_{i}", params))

        # Train the propagator only if enabled
        if self.train_s: 
            tp_params = list(self.target_propagator.parameters())
            if tp_params:
                layerwise_params.append(("target_propagator", tp_params))


        return layerwise_params

    def reset_potential(self):
        for layer in self.network:
            layer.reset()
        self.target_propagator.reset()
        self.trace_in = torch.zeros_like(self.trace_in, device=self.trace_in.device)
        self.trace_in_target = torch.zeros_like(self.trace_in_target, device=self.trace_in_target.device)

    def loss(self, h1, t1, t0):
        dist = (t0.unsqueeze(1) - t0.unsqueeze(0) + 1e-9).pow(2).sum(-1).sqrt()
        h1 = h1.flatten(1)
        t1 = t1.flatten(1)
        y = h1 @ t1.t()  #Logits
        yy = torch.softmax(-dist, dim=-1) # Target
        return self.soft_target_cross_entropy(y, yy),  yy.detach(), torch.softmax(y, dim=-1).detach()

    def soft_target_cross_entropy(self, x, target, reduction='mean'):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean() if reduction == 'mean' else loss

    def detach_membrane_states(self):
        for layer in self.network:
            self._detach_layer_states(layer)
        self._detach_layer_states(self.target_propagator)

    def _detach_layer_states(self, layer):
        if hasattr(layer, 'membrane_potential') and layer.membrane_potential is not None:
            layer.membrane_potential = layer.membrane_potential.detach()
        if hasattr(layer, 'trace') and layer.trace is not None:
            layer.trace = layer.trace.detach()
        if hasattr(layer, 'previous_spike') and layer.previous_spike is not None:
            layer.previous_spike = layer.previous_spike.detach()

class TP_MLP_REG(TP_MLP):

    def _build_network(self, hidden_layers, input_size, output_size, batch_size, snn_params):
        self.network = nn.ModuleList()

        self.network.append(
            self._create_lif_layer(input_size, hidden_layers[0], batch_size, snn_params)
        )

        self.target_propagator = self._create_lif_layer(
            output_size, hidden_layers[0], batch_size, snn_params
        )

        for i in range(1, len(hidden_layers)):
            self.network.append(
                self._create_lif_layer(hidden_layers[i - 1], hidden_layers[i], batch_size * 2, snn_params)
            )

        self.network.append(
            LILayer( input_size=hidden_layers[-1],
                    output_size=output_size,
                    batch_size=batch_size*2,
                    leak="random",
                    train_leak=True,
                    norm=snn_params["norm_type"]                
                    ))
 
    def _create_lif_layer(self, input_size, output_size, batch_size, snn_params):
        return LIFLayer(
            input_size=input_size,
            output_size=output_size,
            batch_size=batch_size,
            vth="random",
            leak_m="random",
            train_leak_m=True,
            leak_t="random",
            train_leak_t=True,
            recurrent=snn_params["l_rec"],
            reset_type=snn_params["l_rst_type"],
            surrogate_type=self.surrogate_type,
            surrogate_scale=self.surrogate_scale,
            norm = snn_params["norm_type"],
        )
    
    def update(self):
        bs = self.ts[0].shape[0]

        # -------- hidden layers -------------------------------------------------
        for i in range(len(self.network) - 1):
            loss, dist_trg, dist_in = self.loss(
                self.trace[i + 1],
                self.trace_target[i + 1],
                self.trace_target[i],
            )

            params = [self.network[i].fc.weight]

            # Check for learnable leak_m and leak_t (e.g. in LIFLayer)
            if hasattr(self.network[i], 'logit_leak_m'):
                params.append(self.network[i].logit_leak_m)
            if hasattr(self.network[i], 'logit_leak_t'):
                params.append(self.network[i].logit_leak_t)

            if self.train_s and i == 0:
                params.append(self.target_propagator.fc.weight)
            if self.network[i].recurrent:
                params.append(self.network[i].rc.weight)

            grads = torch.autograd.grad(loss, params, retain_graph=True)

            k = 0
            self.network[i].fc.weight.grad = grads[k]; k += 1
            if hasattr(self.network[i], 'logit_leak_m'):
                self.network[i].logit_leak_m.grad = grads[k]; k += 1
            if hasattr(self.network[i], 'logit_leak_t'):
                self.network[i].logit_leak_t.grad = grads[k]; k += 1
            if self.train_s and i == 0:
                self.target_propagator.fc.weight.grad = grads[k]; k += 1
            if self.network[i].recurrent:
                self.network[i].rc.weight.grad = grads[k]
