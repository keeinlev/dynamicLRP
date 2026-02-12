#!/bin/bash

# Activate environment if needed. Assuming user runs this within their environment or I can't guess it.
# echo "Starting Captum SQuAD Experiments"

METHODS=("ig" "gradshap" "input_x_gradient" "kernelshap")

# Run T5 experiments
for method in "${METHODS[@]}"; do
    echo "Running T5 with $method..."
    python t5_squad_captum.py --method "$method" > "t5_${method}.log" 2>&1
    echo "Finished T5 $method"
done

# Run RoBERTa experiments
# for method in "${METHODS[@]}"; do
#    echo "Running RoBERTa with $method..."
#    python roberta_squad_captum.py --method "$method" > "roberta_${method}.log" 2>&1
#    echo "Finished RoBERTa $method"
# done

echo "All experiments completed."
