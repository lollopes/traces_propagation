import argparse
from utils import run

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # General run settings
    parser.add_argument("--dataset", type=str, default="CIFAR10DVS", help="Dataset name: CIFAR10DVS or DVSGESTURE")
    parser.add_argument("--algorithm", type=str, default="TP", help="Model: TP (Trace Propagation) or BP (Backprop)")
    parser.add_argument("--run_type", type=str, default="single", choices=["single", "seeds"], help="Run type: single (one run) seeds (10 runs)")
    parser.add_argument("--epoch_print", action="store_true", help="print accuracy at each epoch")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--scheduler_name", type=str, default="CosineAnnealingLR")
    parser.add_argument("--custom_grad", action="store_true")

    # Architecture for MLP
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_layers_size", type=int, default=450)

    # Architecture for CNN
    parser.add_argument("--vgg_variant", type=int, default=9, help="VGG variant (e.g., 9 for VGG9)")

    # Neuron model parameters
    parser.add_argument("--l_vth", type=float, default=0.7)
    parser.add_argument("--l_leak_m", type=float, default=0.11)
    parser.add_argument("--l_leak_t", type=float, default=0.38)
    parser.add_argument("--l_rst_type", type=str, default="Soft")
    parser.add_argument("--l_out_leak_m", type=float, default=1.0)
    parser.add_argument("--l_rec", action="store_true")

    # Training switches
    parser.add_argument("--train_s", action="store_true")

    # Surrogate gradient
    parser.add_argument("--surrogate_type", type=str, default="1")
    parser.add_argument("--surrogate_scale", type=float, default=10.0)

    # Normalization and training behavior
    parser.add_argument("--norm_type", type=str, default="weight")
    parser.add_argument("--layerwise_optim", action="store_true")

    # TSNE plots 
    parser.add_argument("--plot_tsne", action="store_true")

    # Features plots 
    parser.add_argument("--plot_features", action="store_true")
    parser.add_argument("--save_weights", action="store_true")


    args = parser.parse_args()
    run(args)

