###################################
# ### VGG-9 (Trace Propagation) ###
###################################

# python main.py \
#     --dataset DVSGESTURE \
#     --algorithm TP \
#     --run_type single \
#     --epochs 200 \
#     --patience 200 \
#     --optim Adam \
#     --batch_size 64 \
#     --learning_rate 0.0001 \
#     --T 1 \
#     --scheduler_name CosineAnnealingLR \
#     --custom_grad \
#     --vgg_variant 9 \
#     --l_vth 1.0 \
#     --l_leak_m 0.53 \
#     --l_leak_t 0.98 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --norm_type weight \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \
##     --plot_tsne \
##     --plot_features \
##     --save_weights

### EXPECTED RESULTS ###

# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 98.11% (epoch 79) | Final Acc = 94.70%
# Seed 91725: Best Acc = 96.97% (epoch 189) | Final Acc = 95.08%
# Seed 16340: Best Acc = 98.11% (epoch 100) | Final Acc = 95.83%
# Seed 58291: Best Acc = 98.48% (epoch 151) | Final Acc = 96.21%
# Seed 47063: Best Acc = 98.11% (epoch 123) | Final Acc = 96.59%
# Seed 92834: Best Acc = 98.11% (epoch 129) | Final Acc = 94.32%
# Seed 11576: Best Acc = 97.73% (epoch 110) | Final Acc = 95.45%
# Seed 70392: Best Acc = 95.83% (epoch 166) | Final Acc = 94.32%
# Seed 29487: Best Acc = 98.11% (epoch 136) | Final Acc = 95.83%
# Seed 86710: Best Acc = 98.11% (epoch 103) | Final Acc = 97.73%
# ------------------------------------------------------
# Mean Best Test Accuracy: 97.77
# Std Dev of Best Acc:     0.75
# Mean Final Test Accuracy:95.61
# Std Dev of Final Acc:    1.02
# ======================================================

# Top-5 Best Test Accuracies:  [98.10606061 98.10606061 98.10606061 98.10606061 98.48484848]
# Mean of Top-5 Best Acc:      98.18
# Std Dev of Top-5 Best Acc:   0.15

# Top-5 Final Test Accuracies: [95.83333333 95.83333333 96.21212121 96.59090909 97.72727273]
# Mean of Top-5 Final Acc:     96.44
# Std Dev of Top-5 Final Acc:  0.70
# ======================================================


###################################
# ### VGG-9 (Backpropagation-Through-Time) ###
###################################

# python main.py \
#     --dataset DVSGESTURE \
#     --algorithm BP \
#     --run_type single \
#     --epochs 200 \
#     --patience 200 \
#     --optim Adam \
#     --batch_size 64 \
#     --learning_rate 0.001 \
#     --T -1 \
#     --scheduler_name CosineAnnealingLR \
#     --vgg_variant 9 \
#     --l_vth 1.0 \
#     --l_leak_m 0.53 \
#     --l_leak_t 0.5 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --norm_type weight \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \
##     --plot_features \
##     --save_weights


# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 98.48% (epoch 56) | Final Acc = 97.73%
# Seed 91725: Best Acc = 98.48% (epoch 186) | Final Acc = 97.35%
# Seed 16340: Best Acc = 96.97% (epoch 113) | Final Acc = 95.08%
# Seed 58291: Best Acc = 97.73% (epoch 54) | Final Acc = 93.94%
# Seed 47063: Best Acc = 97.73% (epoch 139) | Final Acc = 95.08%
# Seed 92834: Best Acc = 97.73% (epoch 28) | Final Acc = 95.08%
# Seed 11576: Best Acc = 98.48% (epoch 189) | Final Acc = 94.70%
# Seed 70392: Best Acc = 98.86% (epoch 104) | Final Acc = 96.21%
# Seed 29487: Best Acc = 98.48% (epoch 160) | Final Acc = 96.59%
# Seed 86710: Best Acc = 97.73% (epoch 78) | Final Acc = 94.70%
# ------------------------------------------------------
# Mean Best Test Accuracy: 98.07
# Std Dev of Best Acc:     0.55
# Mean Final Test Accuracy:95.64
# Std Dev of Final Acc:    1.19
# ======================================================

# Top-5 Best Test Accuracies:  [98.48484848 98.48484848 98.48484848 98.48484848 98.86363636]
# Mean of Top-5 Best Acc:      98.56
# Std Dev of Top-5 Best Acc:   0.15

# Top-5 Final Test Accuracies: [95.07575758 96.21212121 96.59090909 97.34848485 97.72727273]
# Mean of Top-5 Final Acc:     96.59
# Std Dev of Top-5 Final Acc:  0.93
# ======================================================
