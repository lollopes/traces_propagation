import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.neuron_layers import LILayer, LIFLayerCNN, LIFLayerCNN_PROP

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
    ]
}

class TP_CNN(nn.Module):
    def __init__(self, input_h, input_w, input_channels, output_size, batch_size, snn_params):
        super(TP_CNN, self).__init__()
        self._initialize_params(input_h, input_w, input_channels, output_size, snn_params)
        self._build_network(input_h, input_w, input_channels, output_size, batch_size)
        self._initialize_buffers(batch_size, input_h*input_w*input_channels, output_size)
        self._initialize_buffers_anlaytics()

    def _initialize_buffers_anlaytics(self):

        L = len(self.network)
        # For spike rates
        self.cumulative_spikes = [0 for _ in range(L-1)]
        self.total_spike_neurons = [0 for _ in range(L-1)]

        self.cumulative_spikes_target = [0 for _ in range(L-1)]
        self.total_spike_neurons_target = [0 for _ in range(L-1)]

        # For per-layer loss
        self.loss_sum       = [0.0  for _ in range(L-1)]
        self.loss_count     = 0


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
        self.train_s = snn_params["train_s"]

    def _build_network(
        self, input_h, input_w, input_channels,
        target_dim, batch_size
    ):
        self.network = nn.ModuleList()
        curr_h, curr_w, curr_channels = input_h, input_w, input_channels

        def add_block(out_ch, k, pad, pool_sz, pool_type="max", *, double=False):
            nonlocal curr_h, curr_w, curr_channels
            nxt_h = self._conv_output_size(curr_h, k, pad)
            nxt_w = self._conv_output_size(curr_w, k, pad)

            if pool_type == "aavg":
                nxt_h = self._pool_output_size(nxt_h, pool_sz, adaptive=True)
                nxt_w = self._pool_output_size(nxt_w, pool_sz, adaptive=True)
                pool_kernel = (pool_sz, pool_sz)
            elif pool_sz > 1:
                pool_kernel = (pool_sz, pool_sz)
                nxt_h = self._pool_output_size(nxt_h, pool_sz)
                nxt_w = self._pool_output_size(nxt_w, pool_sz)
            else:
                pool_kernel = (1, 1)

            self.network.append(LIFLayerCNN(
                in_w=curr_w, in_h=curr_h, in_channels=curr_channels,
                out_channels=out_ch,
                batch_size=batch_size * 2 if double else batch_size,  # <- key line
                kernel_size=k, padding=pad,
                pool_size=pool_kernel, pool_type=pool_type,
                vth=self.vth, leak_m=self.leak_m, leak_t=self.leak_t,
                reset_type=self.reset_type,
                surrogate_type=self.surrogate_type,
                surrogate_scale=self.surrogate_scale,
                norm=self.norm_type,
            ))
            curr_h, curr_w, curr_channels = nxt_h, nxt_w, out_ch

        for idx, (out_ch, pool_sz, p_type) in enumerate(VGG_CONFIGS[self.vgg_variant]):
            add_block(out_ch, 3, 1, pool_sz, p_type, double=(idx > 0))

        self.network.append(LILayer(
            input_size=curr_h * curr_w * curr_channels,
            output_size=self.output_size,
            batch_size=batch_size * 2,
            leak=self.leak_m_out,
            norm=self.norm_type
        ))

        self.target_propagator = LIFLayerCNN_PROP(
            in_w=input_w, in_h=input_h, in_channels=input_channels,
            out_channels=VGG_CONFIGS[self.vgg_variant][0][0],
            target_dim=target_dim, batch_size=batch_size,
            kernel_size=3, padding=1, pool_size=(1, 1), pool_type="max",
            leak_m=self.leak_m, leak_t=self.leak_t, reset_type=self.reset_type,
            surrogate_type=self.surrogate_type,
            surrogate_scale=self.surrogate_scale,
            norm=self.norm_type,
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

    def forward(self, x, target):

        self._reset_internal_states()
        bs = x.size(0) 

        x_flat = x.view(bs, -1)
        self.target = target
        self._update_traces(x_flat, target)

        spike_t, _, trace_t = self.target_propagator(target)
        spike_s, _, trace_s = self.network[0](x)

        self._store_internal_states(spike_s, spike_t, trace_s, trace_t)
        self._spike_summer(0, spike_s, spike_t)

        inp_s, inp_t = spike_s.detach(), spike_t.detach()

        for i in range(1, len(self.network) - 1):
            merged = torch.cat((inp_s, inp_t), dim=0)     
            spike_all, _, trace_all = self.network[i](merged)

            spike_s = spike_all[:bs]
            spike_t = spike_all[bs:]
            trace_s = trace_all[:bs]
            trace_t = trace_all[bs:]

            self._store_internal_states(spike_s, spike_t, trace_s, trace_t)
            self._spike_summer(i, spike_s, spike_t)

            inp_s, inp_t = spike_s.detach(), spike_t.detach()

        fc_in = torch.cat((inp_s.flatten(1), inp_t.flatten(1)), dim=0)
        mem_all = self.network[-1](fc_in)
        mem_s, mem_t = mem_all[:bs], mem_all[bs:]

        self.e = F.softmax(mem_s, dim=1) - target

        return mem_s, mem_t

    def inference(self, x):

        self._reset_internal_states()
        bs = x.size(0)

        dummy_t = torch.zeros(bs, self.output_size, device=x.device)
        self._update_traces(x.view(bs, -1), dummy_t)

        spike_s, _, trace_s = self.network[0](x)
        self._store_internal_states(spike_s, None, trace_s, None)
        inp_s = spike_s.detach()

        for i in range(1, len(self.network) - 1):
            merged = torch.cat((inp_s, inp_s), dim=0)          
            spike_all, _, trace_all = self.network[i](merged)
            spike_s = spike_all[:bs]                           
            self._store_internal_states(spike_s, None,
                                        trace_all[:bs], None)
            inp_s = spike_s.detach()

        fc_in   = torch.cat((inp_s.flatten(1), inp_s.flatten(1)), dim=0)
        mem_all = self.network[-1](fc_in)
        mem_s   = mem_all[:bs]
        return mem_s

    def _reset_internal_states(self):
        self.hs = []
        self.ts = []
        self.trace = []
        self.trace_target = []

    def _update_traces(self, x, target):
        B = x.size(0)
        self.trace_in = self._resize_buffer(self.trace_in, B, x.device)
        self.trace_in_target = self._resize_buffer(self.trace_in_target, B, x.device)
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

    def get_layerwise_params(self):
        layerwise_params = []
        for i, layer in enumerate(self.network):
            params = list(layer.parameters())
            if params:  # Only include layers with trainable parameters
                layerwise_params.append((f"layer_{i}", params))

        # Include the target propagator as a separate optimizer group
        if self.train_s:
            params_tp = list(self.target_propagator.parameters())
            if params_tp:
                layerwise_params.append(("target_propagator", params_tp))

        return layerwise_params
    
    def update(self):

        bs = self.ts[0].shape[0]

        loss, dist_trg, dist_in = self.loss(
            self.trace[1],          # student trace
            self.trace_target[1],   # target trace
            self.trace_target[0]
        )

        if self.train_s:
            grad_w = torch.autograd.grad(
                loss, self.target_propagator.fc.weight, retain_graph=True
            )[0]
            self.target_propagator.fc.weight.grad = grad_w

        for i in range(len(self.network) - 1):
            loss, dist_trg, dist_in = self.loss(
                self.trace[i + 1],
                self.trace_target[i + 1],
                self.trace_target[i]
            )
            grad_w = torch.autograd.grad(
                loss, self.network[i].fc.weight, retain_graph=True
            )[0]
            self.network[i].fc.weight.grad = grad_w


        grad_final = torch.einsum('bi,bj->ij', self.e, self.trace[-1].flatten(1)) / bs
        
        self.network[-1].fc.weight.grad = grad_final


    def reset_potential(self):
        for layer in self.network:
            layer.reset()
        self.target_propagator.reset()

        self.trace_in.zero_()
        self.trace_in_target.zero_()

    def loss(self, h1, t1, t0):
        if t0.ndim > 2:
            t0 = t0.flatten(2).permute(2, 0, 1)  
            dist = ((t0.unsqueeze(2) - t0.unsqueeze(1) + 1e-9).pow(2).sum(-1)).sqrt() 
            dist = dist.mean(0) 
        else:
            dist = ((t0.unsqueeze(1) - t0.unsqueeze(0) + 1e-9).pow(2).sum(-1)).sqrt() 

        if h1.ndim > 2:
            h1 = h1.flatten(2).permute(2, 0, 1)  
            t1 = t1.flatten(2).permute(2, 1, 0) 
            y = (h1 @ t1).mean(0)

        else:
            h1 = h1.flatten(1)
            t1 = t1.flatten(1)
            y = h1 @ t1.t()  

        yy = torch.softmax(-dist, dim=-1)  

        return self.soft_target_cross_entropy(y, yy), yy.detach(), torch.softmax(y, dim=-1).detach()

    def _conv_output_size(self, size, kernel_size, padding, stride=1):
        return (size + 2 * padding - kernel_size) // stride + 1

    def _pool_output_size(self, size, pool_size, stride=None, adaptive=False):
        if adaptive:
            return pool_size
        else:
            if stride is None:
                stride = pool_size
            return (size - pool_size) // stride + 1

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

    def _spike_summer(self, idx, spike, spike_targ):

        self.cumulative_spikes[idx] += spike.sum().item()
        self.total_spike_neurons[idx] += spike.numel()
        self.cumulative_spikes_target[idx] += spike_targ.sum().item()
        self.total_spike_neurons_target[idx] += spike_targ.numel()

    def _print_rates(self):

        input_rates = (torch.tensor(self.cumulative_spikes, dtype=torch.float32) / 
                torch.tensor(self.total_spike_neurons, dtype=torch.float32) * 100).round(decimals=2)

        target_rates = (torch.tensor(self.cumulative_spikes_target, dtype=torch.float32) / 
                torch.tensor(self.total_spike_neurons_target, dtype=torch.float32) * 100).round(decimals=2)

        torch.set_printoptions(sci_mode=False, precision=2)
        print(f"input rates: {input_rates}")
        print(f"target rates: {target_rates}")

    def _print_layer_wise_loss(self):
        for idx, layer_loss in enumerate(self.loss_sum):
            print(f"\tLayer {idx} loss: {layer_loss/self.loss_count}")
    
    def _resize_buffer(self,buf, B, device):
        if buf.size(0) != B:
            new_shape = (B, *buf.shape[1:])
            # keep dtype & device
            return torch.zeros(new_shape, dtype=buf.dtype, device=device)
        return buf

    def plot_tsne_maps_epoch(
            self,
            epoch_idx: int,
            total_epochs: int,
            traces_all,                       
            labels_all,                      
            out_dir: str = "plots/tsne",
            n_runs: int = 1,                  
            max_points: int = 2000,          
            random_state: int = 465,
    ):  
        
        if epoch_idx not in (1, total_epochs // 2, total_epochs - 1):
            return

        os.makedirs(out_dir, exist_ok=True)

        labels_np_full = labels_all.cpu().numpy()
        unique_labels  = np.unique(labels_np_full)
        cmap           = plt.get_cmap("tab20", len(unique_labels))

        for layer_idx, X_torch in enumerate(traces_all):
            X = X_torch.detach().cpu().numpy()         
            if X.shape[0] > max_points:
                sel         = np.random.choice(X.shape[0], max_points, replace=False)
                X           = X[sel]
                labels_np   = labels_np_full[sel]
            else:
                labels_np   = labels_np_full

            X = StandardScaler().fit_transform(X)

            fig, axes = plt.subplots(
                1, n_runs, figsize=(4 * n_runs, 4), squeeze=False
            )
            axes = axes.ravel()

            for run_id, ax in enumerate(axes):
                tsne = TSNE(
                    n_components=2,
                    init='pca',       
                    perplexity=50,    
                    learning_rate=100,
                    max_iter=1000,
                    n_iter_without_progress=300,
                    early_exaggeration=12, 
                    verbose=1
                )
                emb = tsne.fit_transform(X)

                for lab in unique_labels:
                    idx = labels_np == lab
                    ax.scatter(
                        emb[idx, 0], emb[idx, 1],
                        s=8, alpha=0.7, color=cmap(int(lab)),
                        label=str(lab) if run_id == 0 else None,
                    )

                ax.set_axis_off()
            # # one legend for the first panel
            # handles, labels_txt = axes[0].get_legend_handles_labels()
            # fig.legend(handles, labels_txt, loc="upper right", frameon=False)

            layer_dir = os.path.join(out_dir, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)

            #fig.suptitle(f"t-SNE – layer {layer_idx} – epoch {epoch_idx}")
            fig.tight_layout(rect=[0, 0, 0.97, 0.95])

            fname = os.path.join(layer_dir, f"epoch_{epoch_idx}.png")
            fig.savefig(fname, dpi=180)
            plt.close(fig)


