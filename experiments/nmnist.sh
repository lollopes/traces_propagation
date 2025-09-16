# #############################
# # ### Feed-Forward (TP) ###
# ############################# 

# python main.py \
#     --dataset NMNIST \
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
#     --hidden_layers_size 200 \
#     --l_vth 1.0 \
#     --l_leak_m 0.98 \
#     --l_leak_t 0.98 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --norm_type none \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \

# ### EXPECTED RESULTS ###

# ================= Seed Sweep Summary =================                                                                                                 
# Seed 38472: Best Acc = 97.27% (epoch 6) | Final Acc = 96.72%                                                                                           
# Seed 91725: Best Acc = 97.18% (epoch 9) | Final Acc = 96.43%                                                                                           
# Seed 16340: Best Acc = 97.21% (epoch 5) | Final Acc = 96.66%                                                                                           
# Seed 58291: Best Acc = 97.33% (epoch 12) | Final Acc = 96.60%                                                                                          
# Seed 47063: Best Acc = 97.31% (epoch 11) | Final Acc = 96.69%                                                                                          
# Seed 92834: Best Acc = 97.44% (epoch 7) | Final Acc = 96.36%                                                                                           
# Seed 11576: Best Acc = 97.12% (epoch 8) | Final Acc = 96.44%                                                                                           
# Seed 70392: Best Acc = 97.32% (epoch 7) | Final Acc = 96.41%                                                                                           
# Seed 29487: Best Acc = 97.24% (epoch 8) | Final Acc = 96.63%                                                                                           
# Seed 86710: Best Acc = 97.18% (epoch 9) | Final Acc = 96.31%                                                                                           
# ------------------------------------------------------                                                                                                 
# Mean Best Test Accuracy: 97.26                                                                                                                         
# Std Dev of Best Acc:     0.09                                                                                                                          
# Mean Final Test Accuracy:96.53                                                                                                                         
# Std Dev of Final Acc:    0.14                                                                                                                          
# ======================================================                                                                                                 
                                                                                                                                                       
# Top-5 Best Test Accuracies:  [97.27 97.31 97.32 97.33 97.44]                                                                                           
# Mean of Top-5 Best Acc:      97.33                                                                                                                     
# Std Dev of Top-5 Best Acc:   0.06                                                                                                                      
                                                                                                                                                       
# Top-5 Final Test Accuracies: [96.6  96.63 96.66 96.69 96.72]                                                                                           
# Mean of Top-5 Final Acc:     96.66                                                                                                                     
# Std Dev of Top-5 Final Acc:  0.04                                                                                                                      
# ======================================================   

# #############################
# # ### Feed-Forward (back-propagation-through-time) ###
# ############################# 

# python main.py \
#     --dataset NMNIST \
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
#     --hidden_layers_size 200 \
#     --l_vth 1.0 \
#     --l_leak_m 0.98 \
#     --l_leak_t 0.98 \
#     --l_rst_type soft \
#     --l_out_leak_m 1 \
#     --norm_type none \
#     --surrogate_type 1 \
#     --surrogate_scale 1 \
#     --layerwise_optim \

# ### EXPECTED RESULTS ###

# ================= Seed Sweep Summary =================
# Seed 38472: Best Acc = 98.43% (epoch 58) | Final Acc = 98.42%
# Seed 91725: Best Acc = 98.31% (epoch 97) | Final Acc = 98.29%
# Seed 16340: Best Acc = 98.43% (epoch 92) | Final Acc = 98.39%
# Seed 58291: Best Acc = 98.43% (epoch 65) | Final Acc = 98.18%
# Seed 47063: Best Acc = 98.33% (epoch 92) | Final Acc = 98.19%
# Seed 92834: Best Acc = 98.36% (epoch 42) | Final Acc = 98.29%
# Seed 11576: Best Acc = 98.52% (epoch 59) | Final Acc = 98.44%
# Seed 70392: Best Acc = 98.36% (epoch 88) | Final Acc = 98.35%
# Seed 29487: Best Acc = 98.32% (epoch 92) | Final Acc = 98.26%
# Seed 86710: Best Acc = 98.42% (epoch 63) | Final Acc = 98.31%
# ------------------------------------------------------
# Mean Best Test Accuracy: 98.39
# Std Dev of Best Acc:     0.06
# Mean Final Test Accuracy:98.31
# Std Dev of Final Acc:    0.08
# ======================================================

# Top-5 Best Test Accuracies:  [98.42 98.43 98.43 98.43 98.52]
# Mean of Top-5 Best Acc:      98.45
# Std Dev of Top-5 Best Acc:   0.04

# Top-5 Final Test Accuracies: [98.31 98.35 98.39 98.42 98.44]
# Mean of Top-5 Final Acc:     98.38
# Std Dev of Top-5 Final Acc:  0.05
#
