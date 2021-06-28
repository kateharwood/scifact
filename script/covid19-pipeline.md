

####################

# Run abstract retrieval.
python3 verisci/inference/abstract_retrieval/vespa.py \
    --dataset data/COVID19/version0516/covidCheck_dev.json \
    --output prediction/pipeline0516_dev/abstract_retrieval_oracle.jsonl \
    --is-oracle True

python3 verisci/inference/abstract_retrieval/vespa.py \
    --dataset data/COVID19/version0516/covidCheck_test.json \
    --cord-corpus data/COVID19/cord19_metadata.csv \
    --output prediction/pipeline0516_test/abstract_retrieval_vespa.jsonl

# Run rationale selection

export CUDA_VISIBLE_DEVICES=3
python -m verisci.inference.rationale_selection.transformer_covid19 \
    --corpus data/COVID19/corpus.json \
    --dataset data/COVID19/version0516/covidCheck_test.json \
    --abstract-retrieval prediction/pipeline0516_test/abstract_retrieval_vespa.jsonl \
    --model  train_output/rationale_roberta_large_feverScifactCovid19/epoch-11-f1-7176 \
    --output-flex prediction/pipeline0516_test/rationale_selection/vespa_feverScifactCovid.jsonl

prediction/pipeline0516_test/abstract_retrieval_vespa.jsonl

python -m verisci.inference.rationale_selection.transformer_covid19 \
    --corpus data/COVID19/corpus.json \
    --dataset data/COVID19/version0516/covidCheck_dev.json \
    --is-oracle True \
    --abstract-retrieval prediction/pipeline0516_dev/abstract_retrieval_oracle.jsonl \
    --output-flex prediction/pipeline0516_dev/rationale_selection/oracle_oracle.jsonl

# trained model
model/rationale_roberta_large_fever_scifact  
train_output/rationale_roberta_large_fever_covid19/epoch-8-f1-6974 
train_output/rationale_roberta_large_covid19/epoch-12-f1-7274
train_output/rationale_roberta_large_feverScifactCovid19/epoch-11-f1-7176

python verisci/evaluate/rationale_selection_covid19.py \
    --dataset data/COVID19/version0516/covidCheck_test.json  \
    --rationale-selection prediction/pipeline0516_test/rationale_selection/oracle_feverScifactCovid.jsonl



# Run label prediction, using the oracle rationales.
export CUDA_VISIBLE_DEVICES=0
python -m verisci.inference.label_prediction.transformer_covid19 \
    --corpus data/COVID19/corpus.json \
    --dataset data/COVID19/version0516/covidCheck_dev.json \
    --rationale-selection prediction/pipeline0516_dev/rationale_selection/oracle_oracle.jsonl \
    --model train_output/label_roberta_large_feverScifactCovid/epoch-10-f1-9045 \
    --mode claim_and_rationale \
    --output prediction/pipeline0516_dev/label_prediction/oracle_oracle_feverScifactCovid.jsonl

# trained model
model/label_roberta_large_fever
train_output/label_roberta_large_covid19/epoch-0-f1-2004
train_output/label_roberta_large_fever_covid19/epoch-10-f1-8982
train_output/label_roberta_large_fever_covid19_UDA/ratio4/thresh0.7_temp0.5_exp_schedule/epoch-13-f1-8994
train_output/label_roberta_large_feverScifactCovid/epoch-10-f1-9045

python verisci/evaluate/label_prediction_covid19.py \
    --dataset data/COVID19/version0516/covidCheck_dev.json  \
    --label-prediction prediction/pipeline0516_dev/label_prediction/oracle_oracle_feverScifactCovid.jsonl


# Run label prediction, using the selected rationales.
export CUDA_VISIBLE_DEVICES=0
python -m verisci.inference.label_prediction.transformer_covid19 \
    --corpus data/COVID19/corpus.json \
    --dataset data/COVID19/version0516/covidCheck_test.json \
    --rationale-selection prediction/pipeline0516_test/rationale_selection/vespa_feverScifactCovid.jsonl \
    --model train_output/label_roberta_large_feverScifactCovid/epoch-10-f1-9045 \
    --mode claim_and_rationale \
    --output prediction/pipeline0516_test/label_prediction/vespa_feverScifactCovid_feverScifactCovid.jsonl

python verisci/evaluate/label_prediction_covid19.py \
    --dataset data/COVID19/version0516/covidCheck_test.json  \
    --label-prediction prediction/pipeline0516_test/label_prediction/oracle_feverCovid_feverScifactCovid.jsonl
