#!/bin/bash
#SBATCH --job-name=dataset_similarity
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=similarity_%j.out
#SBATCH --error=similarity_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu

# Parse command line arguments
OUTPUT_FILE="similarity_results_07_14_extended.png"  # Default output file

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output_file filename.png]"
            exit 1
            ;;
    esac
done

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

# Navigate to the script directory
cd /projects/bfcu/mliu7/all_in_one_pretrainingutil_scripts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env_dev_copy

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Output file: ${OUTPUT_FILE}"
echo "Available GPU memory:"
nvidia-smi

# Run with the Qwen embedding model
python dataset_similarity_fixed.py \
    --method embedding \
    --model "Qwen/Qwen3-Embedding-8B" \
    --sample_size 10000 \
    --use_instruction_format \
    --output_file "${OUTPUT_FILE}"
    
echo "Job completed at $(date)"