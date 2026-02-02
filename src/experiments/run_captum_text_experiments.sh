#!/bin/bash

# Activate environment if needed. Assuming user runs this within their environment.
# echo "Starting Captum Text Experiments"

METHODS=("ig" "gradshap" "input_x_gradient" "kernelshap")

# 1. Run BERT experiments
echo "--- Starting BERT Experiments ---"
for method in "${METHODS[@]}"; do
    echo "Running BERT with $method..."
    python bert_captum.py --method "$method" > "bert_${method}.log" 2>&1
    echo "Finished BERT $method"
done

# 2. Run Llama Wiki experiments
echo "--- Starting Llama Wiki Experiments ---"
cd text/
METHODS_LLAMA_WIKI=("ig" "gradshap" "input_x_gradient" "kernelshap") # Llama scripts currently only support these or need checking
for method in "${METHODS_LLAMA_WIKI[@]}"; do
   echo "Running Llama Wiki with $method..."
   python llama-wiki-captum.py --method "$method" > "llama_wiki_${method}.log" 2>&1
   echo "Finished Llama Wiki $method"
done

# 3. Run Llama IMDB experiments
echo "--- Starting Llama IMDB Experiments ---"
METHODS_LLAMA_IMDB=("input_x_gradient" "kernelshap")
for method in "${METHODS_LLAMA_IMDB[@]}"; do
    echo "Running Llama IMDB with $method..."
    python llama-imdb-captum.py --method "$method" > "llama_imdb_${method}.log" 2>&1
    echo "Finished Llama IMDB $method"
done

echo "All experiments completed."
