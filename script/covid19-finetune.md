
export CUDA_VISIBLE_DEVICES=1

# rationale_selection
## train rationale_selection
### fever
python verisci/training/rationale_selection/transformer_covid19_uda.py \
    --corpus data/COVID19/corpus.json \
    --claim-train data/COVID19/version0620/covidCheck_train.json \
    --claim-dev data/COVID19/version0620/covidCheck_dev.json \
    --model model/rationale_roberta_large_fever \
    --dest train_output/version0620/rationale_roberta_large_fever_covid19 \
    --batch-size-gpu 8
### Covid-check
python verisci/training/rationale_selection/transformer_covid19_uda.py \
    --corpus data/COVID19/corpus.json \
    --claim-train data/COVID19/covidFact_train_20200415.json \
    --claim-dev data/COVID19/covidFact_dev_20200415.json \
    --model roberta-large \
    --dest train_output/rationale_roberta_large_covid19 \
    --batch-size-gpu 2

python verisci/training/rationale_selection/transformer_covid19_uda.py \
    --corpus data/COVID19/corpus.json \
    --claim-train data/COVID19/covidFact_train_20200415.json  \
    --claim-dev data/COVID19/version0516/covidCheck_dev.json  \
    --model model/rationale_roberta_large_fever_scifact \
    --dest train_output/rationale_roberta_large_feverScifactCovid19 \
    --batch-size-gpu 2


## inference rationale_selection on fever-trained climate-fever finetuned roberta large
export CUDA_VISIBLE_DEVICES=1
python verisci/inference/rationale_selection/transformer_covid19.py \
    --corpus data/COVID19/corpus.json \
    --dataset data/COVID19/version0620/covidCheck_dev.json \
    --model model/rationale_roberta_large_fever \
    --output-flex prediction/COVID19_0620/rationale_selection_roberta_large_fever.jsonl

python verisci/inference/rationale_selection/transformer_covid19.py \
    --corpus data/COVID19/corpus.json \
    --dataset data/COVID19/covidFact_dev_20200415.json \
    --model train_output/rationale_roberta_large_fever_covid19/epoch-8-f1-6974 \
    --output-flex prediction/rationale_selection_covid19_finetuned.jsonl


## evaluate rationale_selection on fever-trained climate-fever finetuned roberta large
python verisci/evaluate/rationale_selection_covid19.py \
    --dataset data/COVID19/covidFact_dev_20200415.json  \
    --rationale-selection prediction/rationale_selection_covid19_finetuned.jsonl
    prediction/rationale_selection_covid19_baseline.jsonl
    

## Label Prediction
## covid19 train 
python verisci/training/label_prediction/transformer_fever_covid19_uda.py \
    --corpus data/COVID19/corpus.json \
    --train data/COVID19/covidFact_train_20200415.json \
    --dev data/COVID19/covidFact_dev_20200415.json \
    --model model/label_roberta_large_fever \
    --dest train_output/label_roberta_large_fever_covid19 \
    --batch-size-gpu 2 

python verisci/training/label_prediction/transformer_covid19_uda.py \
    --corpus data/COVID19/corpus.json \
    --train data/COVID19/covidFact_train_20200415.json \
    --dev data/COVID19/covidFact_dev_20200415.json  \
    --model model/label_roberta_large_fever_scifact \
    --dest train_output/label_roberta_large_feverScifactCovid_pick04v \
    --batch-size-gpu 8 


## inference
python -m verisci.inference.label_prediction.transformer_covid19 \
    --corpus data/COVID19/corpus.json \
    --dataset data/COVID19/covidFact_dev_20200415.json \
    --rationale-selection prediction/rationale_selection_covid19_oracle.jsonl \
    --model  train_output/label_roberta_large_fever_covid19/epoch-10-f1-8982 \
    --mode claim_and_rationale \
    --output prediction/label_prediction_covid19_oracle_finetuned.jsonl

 train_output/label_roberta_large_fever_covid19/epoch-10-f1-8982

 prediction/label_prediction_covid19_baseline.jsonl
 prediction/label_prediction_covid19_finetuned.jsonl
 prediction/label_prediction_covid19_oracle_baseline.jsonl
 prediction/label_prediction_covid19_oracle_finetuned.jsonl


## evaluation
python verisci/evaluate/label_prediction_covid19.py \
    --dataset data/COVID19/covidFact_dev_20200415.json  \
    --label-prediction prediction/label_prediction_covid19_oracle_finetuned.jsonl


## UDA train 
batch_size_unsup_ratio="6"
tsa="exp_schedule"
temp="0.4"
thresh="0.9"
python verisci/training/label_prediction/transformer_fever_climate_fever_uda.py \
    --train data/climate_fever/climate-fever-dataset-r1-verisci-train.jsonl \
    --dev data/climate_fever/climate-fever-dataset-r1-verisci-dev.jsonl \
    --data-uda data/UDA_data/semafor/back_translated.jsonl \
    --model model/label_roberta_large_fever \
    --dest train_output/label_roberta_large_fever_climateFever_UDA_ratio6_thresh9 \
    --batch-size-gpu 2 \
    --batch-size-unsup-ratio ${batch_size_unsup_ratio} \
    --uda-softmax-temp ${temp} \
    --uda-confidence-thresh ${thresh} \
    --tsa ${tsa} &> train_output/label_roberta_large_fever_climateFever_UDA_ratio6_thresh9/log.txt
