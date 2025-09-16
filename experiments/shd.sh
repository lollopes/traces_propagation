#!/bin/bash

# Please uncomment the experiment you want to run 

###########################
# ### Feed-Forward (Trace Propagation) ###
###########################

# python main.py \
#     --dataset SHD \
#     --algorithm TP \
#     --custom_grad \
#     --run_type seeds \
#     --epochs 100 \
#     --patience 100 \
#     --optim Adam \
#     --batch_size 128 \
#     --learning_rate 0.0001 \
#     --T 1 \
#     --scheduler_name CosineAnnealingLR \
#     --hidden_layers 1 \
#     --hidden_layers_size 450 \
#     --l_vth 1 \
#     --l_leak_m 0.96 \
#     --l_leak_t 0.97 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --norm_type none \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \

# ### EXPECTED RESULTS ###

# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 64.84% (epoch 81) | Final Acc = 63.47%
# Seed 91725: Best Acc = 67.05% (epoch 81) | Final Acc = 64.34%
# Seed 16340: Best Acc = 63.83% (epoch 86) | Final Acc = 63.28%
# Seed 58291: Best Acc = 65.72% (epoch 88) | Final Acc = 56.16%
# Seed 47063: Best Acc = 67.23% (epoch 97) | Final Acc = 60.57%
# Seed 92834: Best Acc = 65.35% (epoch 84) | Final Acc = 61.58%
# Seed 11576: Best Acc = 61.67% (epoch 82) | Final Acc = 61.67%
# Seed 70392: Best Acc = 63.28% (epoch 86) | Final Acc = 59.24%
# Seed 29487: Best Acc = 68.66% (epoch 92) | Final Acc = 63.60%
# Seed 86710: Best Acc = 66.64% (epoch 65) | Final Acc = 62.32%
# ------------------------------------------------------
# Mean Best Test Accuracy: 65.43
# Std Dev of Best Acc:     1.99
# Mean Final Test Accuracy:61.62
# Std Dev of Final Acc:    2.34
# ======================================================

# Top-5 Best Test Accuracies:  [65.71691176 66.63602941 67.04963235 67.23345588 68.65808824]
# Mean of Top-5 Best Acc:      67.06
# Std Dev of Top-5 Best Acc:   0.96

# Top-5 Final Test Accuracies: [62.31617647 63.28125    63.46507353 63.60294118 64.33823529]
# Mean of Top-5 Final Acc:     63.40
# Std Dev of Top-5 Final Acc:  0.65
# ======================================================

########################
# ### Recurrent (Trace Propagation) ###
########################

# python main.py \
#     --dataset SHD \
#     --algorithm TP \
#     --custom_grad \
#     --run_type seeds \
#     --epochs 100 \
#     --patience 100 \
#     --optim Adam \
#     --batch_size 128 \
#     --learning_rate 0.0001 \
#     --T 1 \
#     --scheduler_name CosineAnnealingLR \
#     --hidden_layers 1 \
#     --hidden_layers_size 450 \
#     --l_vth 0.5 \
#     --l_leak_m 0.85 \
#     --l_leak_t 0.85 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --l_rec \
#     --norm_type none \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \

# ### EXPECTED RESULTS ###

# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 82.26% (epoch 66) | Final Acc = 76.70%
# Seed 91725: Best Acc = 79.69% (epoch 41) | Final Acc = 74.13%
# Seed 16340: Best Acc = 80.74% (epoch 85) | Final Acc = 78.95%
# Seed 58291: Best Acc = 81.76% (epoch 42) | Final Acc = 74.17%
# Seed 47063: Best Acc = 79.46% (epoch 89) | Final Acc = 74.08%
# Seed 92834: Best Acc = 82.49% (epoch 63) | Final Acc = 77.34%
# Seed 11576: Best Acc = 78.54% (epoch 74) | Final Acc = 76.10%
# Seed 70392: Best Acc = 80.10% (epoch 90) | Final Acc = 78.40%
# Seed 29487: Best Acc = 81.16% (epoch 84) | Final Acc = 76.61%
# Seed 86710: Best Acc = 81.34% (epoch 84) | Final Acc = 77.85%
# ------------------------------------------------------
# Mean Best Test Accuracy: 80.75
# Std Dev of Best Acc:     1.22
# Mean Final Test Accuracy:76.43
# Std Dev of Final Acc:    1.71
# ======================================================

# Top-5 Best Test Accuracies:  [81.15808824 81.34191176 81.75551471 82.26102941 82.49080882]
# Mean of Top-5 Best Acc:      81.80
# Std Dev of Top-5 Best Acc:   0.51

# Top-5 Final Test Accuracies: [76.70036765 77.34375    77.84926471 78.40073529 78.95220588]
# Mean of Top-5 Final Acc:     77.85
# Std Dev of Top-5 Final Acc:  0.79
# ======================================================

#############################
# ### Feed-Forward (Backpropagation-Through-Time) ###
#############################

# python main.py \
#     --dataset SHD \
#     --algorithm BP \
#     --run_type seeds \
#     --epochs 100 \
#     --patience 100 \
#     --optim Adam \
#     --batch_size 128 \
#     --learning_rate 0.001 \
#     --T -1 \
#     --scheduler_name CosineAnnealingLR \
#     --hidden_layers 1 \
#     --hidden_layers_size 450 \
#     --l_vth 1 \
#     --l_leak_m 0.96 \
#     --l_leak_t 0.97 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --norm_type none \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \

# ### EXPECTED RESULTS ###

# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 75.60% (epoch 14) | Final Acc = 73.12%
# Seed 91725: Best Acc = 75.41% (epoch 18) | Final Acc = 73.30%
# Seed 16340: Best Acc = 75.28% (epoch 18) | Final Acc = 73.39%
# Seed 58291: Best Acc = 74.95% (epoch 35) | Final Acc = 72.38%
# Seed 47063: Best Acc = 74.68% (epoch 20) | Final Acc = 73.39%
# Seed 92834: Best Acc = 76.10% (epoch 20) | Final Acc = 73.16%
# Seed 11576: Best Acc = 76.24% (epoch 19) | Final Acc = 73.02%
# Seed 70392: Best Acc = 75.97% (epoch 18) | Final Acc = 72.84%
# Seed 29487: Best Acc = 75.23% (epoch 45) | Final Acc = 73.12%
# Seed 86710: Best Acc = 75.32% (epoch 19) | Final Acc = 72.56%
# ------------------------------------------------------
# Mean Best Test Accuracy: 75.48
# Std Dev of Best Acc:     0.48
# Mean Final Test Accuracy:73.03
# Std Dev of Final Acc:    0.32
# ======================================================

# Top-5 Best Test Accuracies:  [75.41360294 75.59742647 75.96507353 76.10294118 76.24080882]                                           
# Mean of Top-5 Best Acc:      75.86
# Std Dev of Top-5 Best Acc:   0.31

# Top-5 Final Test Accuracies: [73.11580882 73.16176471 73.29963235 73.39154412 73.39154412]                                           
# Mean of Top-5 Final Acc:     73.27
# Std Dev of Top-5 Final Acc:  0.11
# ======================================================


#############################
# ### Recurrent (Backpropagation-Through-Time) ###
#############################

# python main.py \
#     --dataset SHD \
#     --algorithm BP \
#     --run_type seeds \
#     --epochs 100 \
#     --patience 100 \
#     --optim Adam \
#     --batch_size 128 \
#     --learning_rate 0.001 \
#     --T -1 \
#     --scheduler_name CosineAnnealingLR \
#     --hidden_layers 1 \
#     --hidden_layers_size 450 \
#     --l_vth 0.5 \
#     --l_leak_m 0.95 \
#     --l_leak_t 0.95 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --l_rec \
#     --norm_type none \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \

# ### EXPECTED RESULTS ###
# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 83.32% (epoch 29) | Final Acc = 81.34%
# Seed 91725: Best Acc = 82.17% (epoch 16) | Final Acc = 79.83%
# Seed 16340: Best Acc = 81.48% (epoch 34) | Final Acc = 79.37%
# Seed 58291: Best Acc = 81.66% (epoch 80) | Final Acc = 80.42%
# Seed 47063: Best Acc = 81.80% (epoch 31) | Final Acc = 80.47%
# Seed 92834: Best Acc = 81.53% (epoch 40) | Final Acc = 80.70%
# Seed 11576: Best Acc = 81.99% (epoch 18) | Final Acc = 81.34%
# Seed 70392: Best Acc = 83.36% (epoch 20) | Final Acc = 80.42%
# Seed 29487: Best Acc = 82.31% (epoch 32) | Final Acc = 82.26%
# Seed 86710: Best Acc = 84.97% (epoch 24) | Final Acc = 81.85%
# ------------------------------------------------------
# Mean Best Test Accuracy: 82.46
# Std Dev of Best Acc:     1.05
# Mean Final Test Accuracy:80.80
# Std Dev of Final Acc:    0.85
# ======================================================

# Top-5 Best Test Accuracies:  [82.16911765 82.30698529 83.31801471 83.36397059 84.97242647]                                           
# Mean of Top-5 Best Acc:      83.23
# Std Dev of Top-5 Best Acc:   1.00

# Top-5 Final Test Accuracies: [80.69852941 81.34191176 81.34191176 81.84742647 82.26102941]                                           
# Mean of Top-5 Final Acc:     81.50
# Std Dev of Top-5 Final Acc:  0.53
# ======================================================

