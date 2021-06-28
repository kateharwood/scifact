
export CUDA_VISIBLE_DEVICES=0

batch_size_unsup_ratio="4"
tsa="exp_schedule"
thresh="0.9"
temp="0.4"

thresh_list="0.6 0.7 0.8 0.9"
temp_list="0.5 0.4"

#cat > train_output/label_roberta_large_fever_covid19_UDA/all_best_result.txt

for temp in $temp_list
do
for thresh in $thresh_list
do
    output="train_output/label_roberta_large_fever_covid19_UDA/ratio${batch_size_unsup_ratio}/thresh${thresh}_temp${temp}_${tsa}"

    python verisci/training/label_prediction/transformer_covid19_uda.py \
        --corpus data/COVID19/corpus.json \
        --train data/COVID19/covidFact_train_20200415.json \
        --dev data/COVID19/covidFact_dev_20200415.json \
        --data-uda data/UDA_data/COVID19/semafor/back_translated_abstract_vespa.jsonl \
        --model model/label_roberta_large_fever \
        --dest ${output} \
        --batch-size-gpu 1 \
        --batch-size-unsup-ratio ${batch_size_unsup_ratio} \
        --uda-softmax-temp ${temp} \
        --uda-confidence-thresh ${thresh} \
        --tsa ${tsa}

    best_model=`cat ${output}/best_model_path.txt`
    echo "$best_model"

    python -m verisci.inference.label_prediction.transformer_covid19 \
        --corpus data/COVID19/corpus.json \
        --dataset data/COVID19/covidFact_dev_20200415.json \
        --rationale-selection prediction/rationale_selection_covid19_oracle.jsonl \
        --model ${best_model} \
        --mode claim_and_rationale \
        --output ${output}/label_prediction.jsonl


    python verisci/evaluate/label_prediction_covid19.py \
        --dataset data/COVID19/covidFact_dev_20200415.json \
        --label-prediction ${output}/label_prediction.jsonl > ${output}/best_eval_result.txt \
        --deleting-model-path ${best_model} 
    
    echo "                         " >> train_output/label_roberta_large_fever_covid19_UDA/all_best_result.txt
    echo "=========================" >> train_output/label_roberta_large_fever_covid19_UDA/all_best_result.txt
    echo "ratio${batch_size_unsup_ratio}_thresh${thresh}_temp${temp}_${tsa}" >> train_output/label_roberta_large_fever_covid19_UDA/all_best_result.txt
    head -n 3 ${output}/best_eval_result.txt >> train_output/label_roberta_large_fever_covid19_UDA/all_best_result.txt

done
done