
from __future__ import annotations
import argparse
import os
from typing import List, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from models.tp_mlp import TP_MLP
from utils import optim_init, seed_init, test as tp_test, train as tp_train

###############################################################################
# Dataset
###############################################################################
class NumpyDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

###############################################################################
# Plotting
###############################################################################
def plot_seed_analysis(
    all_train_acc: List[List[float]],
    all_test_acc: List[List[float]],
    finetune_start: int,
    save_path: str | os.PathLike | None = None,
    show: bool = False,
):
    train_acc = np.stack(all_train_acc)
    test_acc = np.stack(all_test_acc)

    mean_train = train_acc.mean(axis=0)
    std_train = train_acc.std(axis=0)

    mean_test = test_acc.mean(axis=0)
    std_test = test_acc.std(axis=0)

    epochs = np.arange(mean_train.shape[0])
    plt.figure()

    plt.plot(epochs, mean_train, label="Train acc")
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2)

    plt.plot(epochs, mean_test, label="Test acc")
    plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, alpha=0.2)

    plt.axvline(finetune_start - 1.0, linestyle="--", label="Fine-tune start")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

###############################################################################
# Data loader
###############################################################################
def build_dataloaders(save_to, train_batch, nshot, support_batch):
    base_path = os.path.join(save_to, "gsc_v2_data")

    if isinstance(nshot, str) and nshot == "all":
        support_data_file = "support_all_data.npy"
        support_label_file = "support_all_label.npy"
    else:
        support_data_file = f"support_{nshot}shot_data.npy"
        support_label_file = f"support_{nshot}shot_label.npy"

    paths = {
        "train": ("train_data.npy", "train_label.npy"),
        "support": (support_data_file, support_label_file),
        "query": ("query_data.npy", "query_label.npy"),
        "test": ("test_data.npy", "test_label.npy"),
    }

    def make_loader(data_name, label_name, batch_size, shuffle):
        dataset = NumpyDataset(os.path.join(base_path, data_name),
                               os.path.join(base_path, label_name))
        print(f"{data_name}: len={len(dataset)}, requested_bs={batch_size}")
        bs = min(batch_size, len(dataset))
        return DataLoader(dataset, batch_size=bs, shuffle=shuffle)

    return (
        make_loader(*paths["train"], train_batch, shuffle=True),
        make_loader(*paths["support"], support_batch, shuffle=True),
        make_loader(*paths["query"], support_batch, shuffle=False),
        make_loader(*paths["test"], train_batch, shuffle=False),
    )

###############################################################################
# Training loop
###############################################################################
def run(cfg, seed, ft_batch):
    seed_init(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, support_loader, query_loader, test_loader = build_dataloaders(cfg.save_to, cfg.batch, cfg.nshot, ft_batch)
    out_dim = 36

    ######################################################################
    # Phase 1: Fine-tune on support set from scratch (no pretraining)
    ######################################################################
    model = TP_MLP(
        hidden_layers=[450],
        input_size=120,
        output_size=out_dim,
        batch_size=cfg.batch,
        snn_params=dict(
            l_vth=0.66, l_leak_m=0.77, l_leak_t=0.977, l_rec=True,
            l_rst_type="Soft", l_out_leak_m=1.0, train_s=False,
            moving_targets=True, surrogate_type="1", surrogate_scale=10.0,
            seed=seed, norm_type="none", trace_type="spike",
        ),
    ).to(device)

    ft_opts, ft_scheds = optim_init("Adam", cfg.lr, "CosineAnnealingLR", model, cfg.finetune_epochs, layerwise=True)
    for ep in range(cfg.finetune_epochs):
        tp_train(
            model=model,
            train_loader=support_loader,
            device=device,
            output_dim=out_dim,
            optimizers=ft_opts,
            criterion=torch.nn.CrossEntropyLoss(),
            custom_grad=True,
            T=1,
            epoch_idx=ep,
            epochs=cfg.finetune_epochs,
            tsne=False,
            algo_name="TP",
            dataset_name="GSC",
        )
        for sch in ft_scheds.values(): sch.step()

    _, query_acc_finetune_only = tp_test(model, query_loader, device, out_dim, torch.nn.CrossEntropyLoss())

    ######################################################################
    # Phase 2: Train from scratch with full train_loader
    ######################################################################
    model = TP_MLP(
        hidden_layers=[450],
        input_size=120,
        output_size=out_dim,
        batch_size=cfg.batch,
        snn_params=dict(
            l_vth=0.66, l_leak_m=0.77, l_leak_t=0.977, l_rec=True,
            l_rst_type="Soft", l_out_leak_m=1.0, train_s=False,
            moving_targets=True, surrogate_type="1", surrogate_scale=10.0,
            seed=seed, norm_type="none", trace_type="spike",
        ),
    ).to(device)

    optims, scheds = optim_init("Adam", cfg.lr, "CosineAnnealingLR", model, cfg.epochs, layerwise=True)

    tr_hist, te_hist = [], []
    for ep in range(cfg.epochs):
        _, _, tr_acc = tp_train(
            model=model,
            train_loader=train_loader,
            device=device,
            output_dim=out_dim,
            optimizers=optims,
            criterion=torch.nn.CrossEntropyLoss(),
            custom_grad=True,
            T=1,
            epoch_idx=ep,
            epochs=cfg.epochs,
            tsne=False,
            algo_name="TP",
            dataset_name="GSC",
        )
        _, q_acc = tp_test(model, query_loader, device, out_dim, torch.nn.CrossEntropyLoss())
        tr_hist.append(tr_acc)
        te_hist.append(q_acc)
        for sch in scheds.values(): sch.step()

    _, pre_finetune_query_acc = tp_test(model, query_loader, device, out_dim, torch.nn.CrossEntropyLoss())
    _, pre_finetune_test_acc = tp_test(model, test_loader, device, out_dim, torch.nn.CrossEntropyLoss())

    ######################################################################
    # Phase 3: Fine-tune again on support_loader
    ######################################################################
    ft_opts, ft_scheds = optim_init("Adam", cfg.lr, "CosineAnnealingLR", model, cfg.finetune_epochs, layerwise=True)
    for ep in range(cfg.finetune_epochs):
        _, _, ft_acc = tp_train(
            model=model,
            train_loader=support_loader,
            device=device,
            output_dim=out_dim,
            optimizers=ft_opts,
            criterion=torch.nn.CrossEntropyLoss(),
            custom_grad=True,
            T=1,
            epoch_idx=ep,
            epochs=cfg.finetune_epochs,
            tsne=False,
            algo_name="TP",
            dataset_name="GSC",
        )
        _, ft_q_acc = tp_test(model, query_loader, device, out_dim, torch.nn.CrossEntropyLoss())
        tr_hist.append(ft_acc)
        te_hist.append(ft_q_acc)
        for sch in ft_scheds.values(): sch.step()

    _, final_query_acc = tp_test(model, query_loader, device, out_dim, torch.nn.CrossEntropyLoss())
    _, final_test_acc = tp_test(model, test_loader, device, out_dim, torch.nn.CrossEntropyLoss())

    #print(f"Seed {seed} – final query acc after fine-tuning: {final_query_acc:.2f}%, test acc: {final_test_acc:.2f}%")

    return (
        tr_hist, te_hist,
        final_query_acc,
        pre_finetune_query_acc,
        final_test_acc,
        pre_finetune_test_acc,
        query_acc_finetune_only,
    )


###############################################################################
# CLI args
###############################################################################
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed Analysis for TP-MLP on GSC")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-to", type=str, default="/scratch-node/20235438")
    p.add_argument("--ft-batch", type=int, nargs='+', default=[1024],
               help="Batch size(s) used for fine-tuning phase. Can be multiple (e.g. 2 4 8 16).")

    def parse_nshot(val):
        return val if val == "all" else int(val)

    p.add_argument("--nshot", type=parse_nshot, default=5, choices=[1, 5, "all"],
                help="Number of shots for few-shot fine-tuning.")
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    seeds = [928374, 4029138, 17384920, 1234567, 3829104, 789123, 4019283, 992384, 2093841, 6748392]

    for ft_batch in cfg.ft_batch:
        print(f"\n==============================")
        print(f"Running experiments for:")
        print(f"  ➤ k-shot (nshot)             = {cfg.nshot}")
        print(f"  ➤ Fine-tune batch size       = {ft_batch}")
        print(f"==============================\n")

        all_tr_hist, all_te_hist = [], []
        final_q_accs, pre_q_accs = [], []
        final_t_accs, pre_t_accs = [], []
        ft_only_q_accs = []

        for seed in seeds:
            tr_hist, te_hist, final_q, pre_q, final_t, pre_t, ft_only_q = run(cfg, seed, ft_batch)
            all_tr_hist.append(tr_hist)
            all_te_hist.append(te_hist)
            final_q_accs.append(final_q)
            pre_q_accs.append(pre_q)
            final_t_accs.append(final_t)
            pre_t_accs.append(pre_t)
            ft_only_q_accs.append(ft_only_q)

        # Print summary results
        print(f"\n### Summary for fine-tune batch size = {ft_batch}, k-shot = {cfg.nshot} ###")
        print(f"Query accuracy with only fine-tune (no pretraining): {np.mean(ft_only_q_accs):.2f} ± {np.std(ft_only_q_accs):.2f}%")
        print(f"Query accuracy before fine-tuning (after pretrain):  {np.mean(pre_q_accs):.2f} ± {np.std(pre_q_accs):.2f}%")
        print(f"Final query accuracy after fine-tuning:              {np.mean(final_q_accs):.2f} ± {np.std(final_q_accs):.2f}%")
        print(f"Test accuracy before fine-tuning:                   {np.mean(pre_t_accs):.2f} ± {np.std(pre_t_accs):.2f}%")
        print(f"Final test accuracy after fine-tuning:              {np.mean(final_t_accs):.2f} ± {np.std(final_t_accs):.2f}%")


