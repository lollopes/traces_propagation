import torch 
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os 

# Plottings import
import matplotlib.pyplot as plt

# Models import 
from models.tp_mlp import TP_MLP
from models.tp_cnn import TP_CNN
from models.bp_cnn import BP_CNN
from models.bp_mlp import BP_MLP    

# Datasets import 
from datasets.CIFAR10DVS.dataset_init import dataset_init as CIFAR10DVS_INIT
from datasets.DVSGESTURE.dataset_init import dataset_init as DVSGESTURE_INIT
from datasets.NMNIST.dataset_init import dataset_init as NMNIST_INIT
from datasets.SHD.dataset_init import dataset_init as SHD_INIT

seeds = [38472, 91725, 16340, 58291, 47063 , 92834, 11576, 70392, 29487, 86710]

def run(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")
    print_args(args)

    seeds_to_run = [seeds[0]] if args.run_type == "single" else seeds
    best_final_test_acc = []
    best_test_acc = []
    best_epochs = []

    for seed_idx, seed in enumerate(seeds_to_run):
        print(f"\n--- Running seed {seed} ({seed_idx + 1}/{len(seeds_to_run)}) ---\n")
        
        # Initialize the seed
        seed_init(seed)

        # Initialize the dataset 
        dataset_params = dataset_init(dataset_name=args.dataset,
                                      batch_size=args.batch_size)
        train_loader = dataset_params[0]
        test_loader = dataset_params[1]
        out_dim = dataset_params[2]

        model_params = model_params_init(args)

        # Initialize the model 
        model = model_init(algorithm=args.algorithm,
                           dataset=args.dataset,
                           model_params=model_params,
                           dataset_params=dataset_params,
                           batch_size=args.batch_size,
                           device=device)
        
        # Output layer loss 
        criterion = torch.nn.CrossEntropyLoss()

        # Create optimizers and schedulers
        optimizers, schedulers = optim_init(optimizer_name=args.optim,
                                            learning_rate=args.learning_rate,
                                            scheduler_name=args.scheduler_name,
                                            model=model,
                                            epochs=args.epochs,
                                            layerwise=args.layerwise_optim)

        best_test_accuracy = 0.0
        best_epoch = -1

        for epoch in tqdm(range(args.epochs), desc=f"Seed {seed}", dynamic_ncols=True):
            train_loss, train_loss_t, train_accuracy = train(
                model=model,
                train_loader=train_loader,
                device=device,
                output_dim=out_dim,
                optimizers=optimizers,
                criterion=criterion,
                custom_grad=(args.algorithm == "TP"),
                T=args.T,
                epoch_idx=epoch,
                epochs=args.epochs,
                tsne=args.plot_tsne,
                algo_name=args.algorithm,
                dataset_name=args.dataset,
            )

            if schedulers:
                for sched in schedulers.values():
                    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sched.step(train_accuracy)
                    else:
                        sched.step()

            test_loss, test_accuracy = test(
                model=model,
                test_loader=test_loader,
                device=device,
                output_dim=out_dim,
                criterion=criterion
            )

            # Update best accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_epoch = epoch

            # Print per epoch
            if args.epoch_print:
                tqdm.write(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f} | "
                        f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f} | Best Test Acc: {best_test_accuracy:.2f}")


        best_test_acc.append(best_test_accuracy)
        best_final_test_acc.append(test_accuracy)
        best_epochs.append(best_epoch)

        # Plot features 
        if args.plot_features:
            plot_features(model, test_loader, device, algo_name=args.algorithm, dataset_name=args.dataset)

        if args.save_weights:
            save_weights(model,args.algorithm, args.dataset)


    # Final summary 
    if args.run_type == "seeds":
        best_test_np = np.array(best_test_acc)
        final_test_np = np.array(best_final_test_acc)
        best_epochs_np = np.array(best_epochs)

        print("\n================= Seed Sweep Summary =================")
        for i, seed in enumerate(seeds_to_run):
            print(f"Seed {seed}: "
                f"Best Acc = {best_test_np[i]:.2f}% (epoch {best_epochs_np[i]}) | "
                f"Final Acc = {final_test_np[i]:.2f}%")

        print("------------------------------------------------------")
        print(f"Mean Best Test Accuracy: {best_test_np.mean():.2f}")
        print(f"Std Dev of Best Acc:     {best_test_np.std():.2f}")
        print(f"Mean Final Test Accuracy:{final_test_np.mean():.2f}")
        print(f"Std Dev of Final Acc:    {final_test_np.std():.2f}")
        print("======================================================")

        # Top-5 stats
        top5_best = np.sort(best_test_np)[-5:]
        top5_final = np.sort(final_test_np)[-5:]

        print(f"\nTop-5 Best Test Accuracies:  {top5_best}")
        print(f"Mean of Top-5 Best Acc:      {top5_best.mean():.2f}")
        print(f"Std Dev of Top-5 Best Acc:   {top5_best.std():.2f}")
        print(f"\nTop-5 Final Test Accuracies: {top5_final}")
        print(f"Mean of Top-5 Final Acc:     {top5_final.mean():.2f}")
        print(f"Std Dev of Top-5 Final Acc:  {top5_final.std():.2f}")
        print("======================================================")

def plot_features(model, test_loader, device, *, max_channels=16, algo_name=None, dataset_name=None):

    save_root = os.path.join("plots", "features", algo_name, dataset_name)
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    images, labels = next(iter(test_loader))    
    x     = images[0:1].to(device)              
    label = labels[0].item()                   
    T     = x.shape[1]

    input_rate = x.cpu().mean(dim=1)[0]         

    if input_rate.shape[0] == 1:                 
        vis_img = input_rate[0]
    else:                                        
        vis_img = input_rate.mean(dim=0)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(vis_img, cmap="gray")
    ax.axis("off")
    ax.set_title(f"Input spike‑rate  |  label {label}", fontsize=10)
    fig.tight_layout()

    # save alongside the hidden‑layer plots
    fname = os.path.join(save_root, "input.png")
    fig.savefig(fname, dpi=300)
    print(f"Saved → {fname}")

    plt.show()
    plt.close(fig)                         

    feat_acc, handles = defaultdict(list), []

    def mk_hook(layer_idx):
        def _hook(_, __, out):
            feat_acc[layer_idx].append(out[0][0].detach().cpu())  # (C,H,W)
        return _hook

    for idx, layer in enumerate(model.network[:-1]):
        handles.append(layer.register_forward_hook(mk_hook(idx)))

    model.reset_potential()
    with torch.no_grad():
        for t in range(T):
            _ = model.inference(x[:, t])

    for h in handles:
        h.remove()

    for ℓidx, stack in feat_acc.items():
        mean_maps = torch.stack(stack).mean(0).numpy()  
        C, H, W    = mean_maps.shape
        n_show     = min(max_channels, C)
        n_cols     = 4
        n_rows     = int(np.ceil(n_show / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.6),
            squeeze=False
        )

        for i, ax in enumerate(fig.axes):
            ax.axis("off")
            if i < n_show:
                ax.imshow(mean_maps[i], cmap="gray")
                ax.set_title(f"Ch {i}", fontsize=8)

        fig.suptitle(f"Layer {ℓidx}  |  true label = {label}", fontsize=12)
        fig.tight_layout()

        fname = os.path.join(save_root, f"{ℓidx}.png")
        fig.savefig(fname, dpi=300)
        print(f"Saved → {fname}")

        plt.show()     
        plt.close(fig)

def save_weights(model, algo_name, dataset_name, save_dir="saved_weights"):
    os.makedirs(save_dir, exist_ok=True)
    
    for name, p in model.named_parameters():
        if p.requires_grad and p.ndim >= 2:
            flat_tensor = p.detach().cpu().flatten()
            filename = f"{algo_name}_{dataset_name}_{name.replace('.', '_')}.pt"
            path = os.path.join(save_dir, filename)
            torch.save(flat_tensor, path)

def print_args(args):
    """
    Print only relevant parameters based on dataset type (CNN or MLP).
    """
    cnn_datasets = {"CIFAR10DVS", "DVSGESTURE"}
    mlp_datasets = {"SHD", "NMNIST"}

    shared_keys = {
        "dataset", 
        "algorithm", 
        "run_type", 
        "save_best",
        "seed", 
        "epochs", 
        "patience", 
        "optim", 
        "batch_size",
        "learning_rate", 
        "T", 
        "scheduler_name",
        "l_vth", 
        "l_leak_m", 
        "l_leak_t", 
        "l_rst_type", 
        "l_out_leak_m",
        "train_s", 
        "surrogate_type", 
        "surrogate_scale",
        "norm_type", 
        "layerwise_optim",
        "custom_grad",
        "plot_tsne",
        "plot_features",
        "epoch_print",
        "save_weights"
    }

    if args.dataset in cnn_datasets:
        specific_keys = {"vgg_variant"}
    elif args.dataset in mlp_datasets:
        specific_keys = {"hidden_layers", "hidden_layers_size", "l_rec"}
    else:
        specific_keys = set()

    all_keys = shared_keys.union(specific_keys)

    print("========= Running with the following parameters =========")
    for key in sorted(all_keys):
        value = getattr(args, key, None)
        print(f"{key:20}: {value}")
    print("=========================================================")

def model_params_init(args):
    model_params = {
        "l_vth": args.l_vth,
        "l_leak_m": args.l_leak_m,
        "l_leak_t": args.l_leak_t,
        "l_rst_type": args.l_rst_type,
        "l_out_leak_m": args.l_out_leak_m,
        "surrogate_type": args.surrogate_type,
        "surrogate_scale": args.surrogate_scale,
        "norm_type": args.norm_type,
        "train_s": args.train_s
    }

    if args.dataset in {"SHD", "NMNIST"}:  # MLP datasets
        model_params["hidden_layers"] = args.hidden_layers
        model_params["hidden_layers_size"] = args.hidden_layers_size
        model_params["l_rec"] = args.l_rec

    elif args.dataset in {"CIFAR10DVS", "DVSGESTURE"}:  # CNN datasets
        model_params["vgg_variant"] = args.vgg_variant

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return model_params

def optim_init(optimizer_name, learning_rate, scheduler_name, model, epochs, layerwise):
    layerwise_params = model.get_layerwise_params()

    if layerwise:
        # === Multiple optimizers ===
        if optimizer_name == "Adam":
            optimizers = {
                name: torch.optim.Adam(params, lr=learning_rate)
                for name, params in layerwise_params
            }
        elif optimizer_name == "SGD":
            optimizers = {
                name: torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
                for name, params in layerwise_params
            }
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # === Multiple schedulers ===
        if scheduler_name == "CosineAnnealingLR":
            schedulers = {
                name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=learning_rate * 0.25)
                for name, opt in optimizers.items()
            }
        elif scheduler_name == "StepLR":
            schedulers = {
                name: torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
                for name, opt in optimizers.items()
            }
        elif scheduler_name == "ReduceLROnPlateau":
            schedulers = {
                name: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
                for name, opt in optimizers.items()
            }
        else:
            schedulers = None

    else:
        # === Single optimizer ===
        all_params = [param for _, params in layerwise_params for param in params]
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(all_params, lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # === Single scheduler ===
        if scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.25)
        elif scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        else:
            scheduler = None

        optimizers = {"global": optimizer}
        schedulers = {"global": scheduler}

    return optimizers, schedulers

def dataset_init(dataset_name, batch_size):

    if dataset_name == "CIFAR10DVS":
        return CIFAR10DVS_INIT(batch_size=batch_size)

    elif dataset_name == "DVSGESTURE":
        return DVSGESTURE_INIT(batch_size=batch_size)

    elif dataset_name == "NMNIST":
        return NMNIST_INIT(batch_size=batch_size)

    elif dataset_name == "SHD":
        return SHD_INIT(batch_size=batch_size)
    
    else:
        return None

def model_init(algorithm, dataset, model_params, dataset_params, batch_size, device):

    cnn_datasets = {"CIFAR10DVS", "DVSGESTURE"}
    mlp_datasets = {"SHD", "NMNIST"}

    if dataset in cnn_datasets:
        out_dim, in_w, in_h, in_ch = dataset_params[2], dataset_params[3], dataset_params[4], dataset_params[5]
        algo_type = "CNN"
    elif dataset in mlp_datasets:
        out_dim, in_dim = dataset_params[2], dataset_params[3]
        hidden_layers = [model_params["hidden_layers_size"] for i in range(model_params["hidden_layers"])]
        algo_type = "MLP"

    # CNN case  
    if algo_type == "CNN":
        if algorithm == "TP":
            model = TP_CNN(
                input_h=in_h, 
                input_w=in_w, 
                input_channels=in_ch,
                output_size=out_dim,
                batch_size=batch_size,
                snn_params=model_params,
            ).to(device)

        elif algorithm == "BP":
            model = BP_CNN(
                            input_h=in_h, 
                            input_w=in_w, 
                            input_channels=in_ch,
                            output_size=out_dim,
                            batch_size=batch_size,
                            snn_params=model_params,
                          ).to(device)

    # MLP case
    elif algo_type == "MLP":
        if algorithm == "TP":
            model = TP_MLP(
                            hidden_layers=hidden_layers,
                            input_size=in_dim,
                            output_size=out_dim,
                            batch_size=batch_size,
                            snn_params=model_params 
                        ).to(device)
        elif algorithm == "BP":
            model = BP_MLP(
                            hidden_layers=hidden_layers,
                            input_size=in_dim,
                            output_size=out_dim,
                            batch_size=batch_size,
                            snn_params=model_params 
                        ).to(device)

    return model 

def seed_init(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model,
          train_loader, 
          device, 
          output_dim, 
          optimizers, 
          criterion,
          custom_grad,
          T,
          epoch_idx,
          epochs,
          tsne,
          algo_name,
          dataset_name,
          ):

    if tsne:
        labels_epoch = []
        n_layers  = len(model.network) - 1
        traces_ep = [[] for _ in range(n_layers)]

    batch_losses = []
    batch_losses_t = []
    correct = 0
    total = 0
    model.train()

    for images, labels in train_loader:
        model.reset_potential()
        input, labels = images.to(device), labels.to(device)
        labels_one_hot = F.one_hot(labels, output_dim).float()

        x = input

        # Zeros the optimizers
        for opt in optimizers.values():
            opt.zero_grad()

        time_steps = x.shape[1]

        for t in range(time_steps):  # Loop over time  steps
            x_t = x[:, t, :]
            out_t, out_teacher_t = model(x_t, labels_one_hot) 
            loss_t = criterion(out_t, labels)
            loss_teacher_t = criterion(out_teacher_t, labels)
            
            if custom_grad:
                if T == 1: # Update at each time step
                    for opt in optimizers.values():
                        opt.zero_grad()

                    model.update()

                    for opt in optimizers.values():
                        opt.step()
                                        
                else: # Update summed across time steps
                    model.update()
                model.detach_membrane_states() 
        
        if tsne:
            labels_epoch.append(labels)
            B = labels.size(0)
            for l in range(n_layers):
                z = model.trace[l+1][:B]             
                traces_ep[l].append(z.detach().view(z.size(0), -1).cpu())
        
        # Final update (mostly for BP)
        if T == -1 and not custom_grad:
            #breakpoint()
            loss_t.backward()
            for opt in optimizers.values():
                opt.step()

        # Compute loss and accuracy
        batch_losses.append(loss_t.item())
        batch_losses_t.append(loss_teacher_t.item())
        _, predicted = torch.max(out_t.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        
    # Compute epoch metrics
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_loss_t = sum(batch_losses_t) / len(batch_losses)
    epoch_accuracy = 100 * correct / total

    if tsne:
        for l in range(n_layers):
            traces_ep[l] = torch.cat(traces_ep[l], 0) 
        labels_epoch = torch.cat(labels_epoch)
        model.plot_tsne_maps_epoch(         
            epoch_idx    = epoch_idx,
            total_epochs = epochs,
            traces_all   = traces_ep,
            labels_all   = labels_epoch,
            out_dir      = f"plots/tsne/{algo_name}/{dataset_name}"
        )

    return epoch_loss, epoch_loss_t, epoch_accuracy

def test(model, 
         test_loader,
         device,
         output_dim,
         criterion):

    device = device
    model.eval()  # Set the model to evaluation mode
    test_losses = []
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed    
        for images, labels in test_loader:

            model.reset_potential()

            input, labels = images.to(device), labels.to(device)
            # Encode labels
            labels_one_hot = F.one_hot(labels, output_dim).float()

            # Temporal loop
            x = input 
            for t in range(x.shape[1]):
                x_t = x[:,t,:]
                out_t = model.inference(x_t)
            
            output = F.softmax(out_t, dim=1)
            
            # Calculate loss and add to list
            test_losses.append(criterion(output, labels).item())
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels_one_hot.argmax(dim=1)).sum().item()

    # Average loss for this epoch
    test_loss = sum(test_losses) / len(test_losses)
    # Average accuracy for this epoch
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy
