###################################
# ### VGG-9 (Trace Propagation) ###
###################################

# python main.py \
#     --dataset CIFAR10DVS \
#     --algorithm TP \
#     --run_type seeds \
#     --epochs 300 \
#     --patience 300 \
#     --optim Adam \
#     --batch_size 64 \
#     --learning_rate 0.0001 \
#     --T 1 \
#     --scheduler_name CosineAnnealingLR \
#     --custom_grad \
#     --vgg_variant 9 \
#     --l_vth 0.5 \
#     --l_leak_m 0.18 \
#     --l_leak_t 0.19 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --norm_type weight \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \
#     --plot_tsne

# Golden Results # 
# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 71.88% (epoch 244) | Final Acc = 71.46%
# Seed 91725: Best Acc = 71.46% (epoch 243) | Final Acc = 68.75%
# Seed 16340: Best Acc = 70.94% (epoch 240) | Final Acc = 69.17%
# Seed 58291: Best Acc = 71.25% (epoch 256) | Final Acc = 69.90%
# Seed 47063: Best Acc = 71.25% (epoch 274) | Final Acc = 68.54%
# Seed 92834: Best Acc = 71.46% (epoch 233) | Final Acc = 69.90%
# Seed 11576: Best Acc = 70.73% (epoch 295) | Final Acc = 67.81%
# Seed 70392: Best Acc = 70.73% (epoch 246) | Final Acc = 70.00%
# Seed 29487: Best Acc = 70.83% (epoch 272) | Final Acc = 67.92%
# Seed 86710: Best Acc = 72.08% (epoch 264) | Final Acc = 70.10%
# ------------------------------------------------------
# Mean Best Test Accuracy: 71.26
# Std Dev of Best Acc:     0.45
# Mean Final Test Accuracy:69.35
# Std Dev of Final Acc:    1.07
# ======================================================

# Top-5 Best Test Accuracies:  [71.25       71.45833333 71.45833333 71.875      72.08333333]
# Mean of Top-5 Best Acc:      71.62
# Std Dev of Top-5 Best Acc:   0.31

# Top-5 Final Test Accuracies: [69.89583333 69.89583333 70.         70.10416667 71.45833333]
# Mean of Top-5 Final Acc:     70.27
# Std Dev of Top-5 Final Acc:  0.60
# ======================================================
