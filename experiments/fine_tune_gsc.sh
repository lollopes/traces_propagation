python gsc/fine_tune_gsc.py --nshot 1
python gsc/fine_tune_gsc.py --nshot 5
python gsc/fine_tune_gsc.py --nshot "all"

# EXPECTED RESULTS #
# Number of shots: 1
# Seed 2133 – final query acc after fine-tuning: 93.24%
# Seed 43545 – final query acc after fine-tuning: 90.54%
# Seed 31268 – final query acc after fine-tuning: 89.19%
# Seed 907965 – final query acc after fine-tuning: 89.19%
# Seed 3425 – final query acc after fine-tuning: 89.19%
# Seed 76798 – final query acc after fine-tuning: 91.89%
# Seed 7234 – final query acc after fine-tuning: 94.59%
# Seed 9874 – final query acc after fine-tuning: 94.59%
# Seed 56654 – final query acc after fine-tuning: 93.24%
# Seed 17773 – final query acc after fine-tuning: 90.54%
# Query accuracy before fine-tuning (mean ± std): 81.62 ± 2.20%
# Final accuracy mean ± std over seeds: 91.62 ± 2.08%
# Number of shots: 5
# Seed 2133 – final query acc after fine-tuning: 95.95%
# Seed 43545 – final query acc after fine-tuning: 93.24%
# Seed 31268 – final query acc after fine-tuning: 94.59%
# Seed 907965 – final query acc after fine-tuning: 91.89%
# Seed 3425 – final query acc after fine-tuning: 93.24%
# Seed 76798 – final query acc after fine-tuning: 95.95%
# Seed 7234 – final query acc after fine-tuning: 93.24%
# Seed 9874 – final query acc after fine-tuning: 94.59%
# Seed 56654 – final query acc after fine-tuning: 91.89%
# Seed 17773 – final query acc after fine-tuning: 89.19%
# Query accuracy before fine-tuning (mean ± std): 81.62 ± 2.20%
# Final accuracy mean ± std over seeds: 93.38 ± 1.95%
# Number of shots: all
# Seed 2133 – final query acc after fine-tuning: 97.30%
# Seed 43545 – final query acc after fine-tuning: 97.30%
# Seed 31268 – final query acc after fine-tuning: 94.59%
# Seed 907965 – final query acc after fine-tuning: 98.65%
# Seed 3425 – final query acc after fine-tuning: 97.30%
# Seed 76798 – final query acc after fine-tuning: 97.30%
# Seed 7234 – final query acc after fine-tuning: 97.30%
# Seed 9874 – final query acc after fine-tuning: 98.65%
# Seed 56654 – final query acc after fine-tuning: 97.30%
# Seed 17773 – final query acc after fine-tuning: 97.30%
# Query accuracy before fine-tuning (mean ± std): 81.62 ± 2.20%
# Final accuracy mean ± std over seeds: 97.30 ± 1.05%
