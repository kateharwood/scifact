
export CUDA_VISIBLE_DEVICES=1,2

batch_size_unsup_ratio="8"
tsa="exp_schedule"
thresh="0.9"
temp="0.4"

thresh_list="0.7 0.8 0.9"
temp_list="0.4 0.5 0.6"


for temp in $temp_list
do
for thresh in $thresh_list
do
    output="train_output/version0620/rationale_roberta_large_fever_covid19_UDA/ratio${batch_size_unsup_ratio}/thresh${thresh}_temp${temp}_${tsa}"

    python verisci/training/rationale_selection/transformer_covid19_uda.py \
        --corpus data/COVID19/corpus.json \
        --claim-train data/COVID19/version0620/covidCheck_train.json \
        --claim-dev data/COVID19/version0620/covidCheck_dev.json  \
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

    python -m verisci.inference.rationale_selection.transformer_covid19 \
        --corpus data/COVID19/corpus.json \
        --dataset data/COVID19/version0620/covidCheck_dev.json  \
        --abstract-retrieval prediction/pipeline0620_dev/abstract_retrieval_oracle.jsonl \
        --model ${best_model} \
        --output-flex ${output}/rationale_selection_dev.jsonl


    python verisci/evaluate/rationale_selection_covid19.py \
        --dataset data/COVID19/version0620/covidCheck_dev.json \
        --rationale-selection ${output}/rationale_selection_dev.jsonl > ${output}/best_eval_result.txt \
        --deleting-model-path ${best_model} \
        --deleting-model-threshhold 0.64
    
    echo "                         " >> train_output/version0620/rationale_roberta_large_fever_covid19_UDA/all_best_result.txt
    echo "=========================" >> train_output/version0620/rationale_roberta_large_fever_covid19_UDA/all_best_result.txt
    echo "ratio${batch_size_unsup_ratio}_thresh${thresh}_temp${temp}_${tsa}" >> train_output/version0620/rationale_roberta_large_fever_covid19_UDA/all_best_result.txt
    head -n 3 ${output}/best_eval_result.txt >> train_output/version0620/rationale_roberta_large_fever_covid19_UDA/all_best_result.txt

done
done