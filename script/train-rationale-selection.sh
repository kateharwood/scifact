

# Create a prediction folder to store results.
model="roberta-large"
batch_size_unsup_ratio="0"
uda="_uda"
#rm -rf train_output/rationale_selection/transformer_scifact${uda}${batch-size-unsup-ratio}/${model}
mkdir -p train_output/rationale_selection/transformer_scifact${uda}_de${batch_size_unsup_ratio}/${model}

echo "Training"
# python3 verisci/training/rationale_selection/transformer_scifact.py \
#     --claim-train data/claims_train.jsonl \
#     --claim-dev data/claims_dev.jsonl \
#     --corpus data/corpus.jsonl \
#     --model ${model} \
#     --dest train_output

#export CUDA_VISIBLE_DEVICES=0
python3 verisci/training/rationale_selection/transformer_scifact_uda.py \
    --claim-train data/claims_train.jsonl \
    --claim-dev data/claims_dev.jsonl \
    --claim-unsup data/claims_train.jsonl \
    --claim-aug data/claims_train_de.jsonl \
    --corpus data/corpus.jsonl \
    --model ${model} \
    --batch-size-gpu 8 \
    --dest "train_output/rationale_selection/transformer_scifact${uda}_de${batch_size_unsup_ratio}/${model}" \
    --batch-size-unsup-ratio ${batch_size_unsup_ratio} 
    #&> train_output/rationale_selection/transformer_scifact${uda}${batch-size-unsup-ratio}/${model}/terminal_log.txt