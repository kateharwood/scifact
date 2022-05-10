import argparse
import jsonlines

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm

#from verisci.inference.label_prediction.specificity import get_specificity

parser = argparse.ArgumentParser()
parser.add_argument('--no-rationales', type=bool, default=False, required=False) # If we send the full abstract through as the rationale, not sentences from it
parser.add_argument('--corpus', type=str, required=False)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--mode', type=str, default='claim_and_rationale', choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--include_nouns', type=bool, default=False )
parser.add_argument('--fuzzy', type=bool, default=False)
parser.add_argument('--specificity_thres', type=float, default=None)


args = parser.parse_args()

print(args.mode)

corpus = {doc['cord_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)
output = jsonlines.open(args.output, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).eval().to(device)

# Testing other BERT-based models
# tokenizer = AutoTokenizer.from_pretrained("stanleychu2/roberta-fever", config=AutoConfig.from_pretrained('stanleychu2/roberta-fever'))
# config = AutoConfig.from_pretrained("stanleychu2/roberta-fever/config.json")
# model = AutoModelForSequenceClassification.from_pretrained("stanleychu2/roberta-fever").eval().to(device)

# tokenizer = AutoTokenizer.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', config=AutoConfig.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'))
# model = RobertaForSequenceClassification.from_pretrained('ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', num_labels =3, return_dict=True).eval().to(device)#ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli


LABELS = ['REFUTE', 'NOTENOUGHINFO', 'SUPPORT']

def encode(sentences, claims):
    text = {
        "claim_and_rationale": list(zip(sentences, claims)),
        "only_claim": claims,
        "only_rationale": sentences
    }[args.mode]
    encoded_dict = tokenizer.batch_encode_plus(
        text,
        pad_to_max_length=True,
        return_tensors='pt'
    )
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            text,
            max_length=512,
            pad_to_max_length=True,
            truncation_strategy='only_first',
            return_tensors='pt'
        )
    encoded_dict = {key: tensor.to(device)
                  for key, tensor in encoded_dict.items()}
    return encoded_dict


with torch.no_grad():
    for data, selection in tqdm(list(zip(dataset, rationale_selection))):
        if not args.no_rationales:
            assert data['id'] == selection['claim_id']

        claim = data['claim']
        results = {}

        if not selection['evidence']: # deal with non-claim
            results["000"] = {'label': 'NOTENOUGHINFO', 'confidence': 1}

        for doc_id, indices in selection['evidence'].items():
            if not indices:
                results[doc_id] = {'label': 'NOTENOUGHINFO', 'confidence': 1}
            else:
                if "abstracts" in selection and selection["abstracts"]:
                    evidence = ' '.join([selection["abstracts"][doc_id][i] for i in indices])
                else:
                    # Testing contextualized rationales: adding sentence on either side of indexed sentence
#                     multi_indices = []
#                     for i in indices:
#                         if i == 0:
#                             multi_indices.extend([i, i+1])
#                         elif i == len(corpus[doc_id]['abstract'])-1:
#                             multi_indices.extend([i-1, i])
#                         else:
#                             multi_indices.extend([i-1, i, i+1])
#                    if multi_indices:
#                         indices = multi_indices

                    # Testing sending all the sentences in the abstract through the model
                    if args.no_rationales:
                        evidence = ' '.join(corpus[doc_id]['abstract'])
                    else:
                        evidence = ' '.join([corpus[doc_id]['abstract'][i] for i in indices])
                encoded_dict = encode([evidence], [claim])
                label_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[0]
                label_index = label_scores.argmax().item()
                label_confidence = label_scores[label_index].item()
                results[doc_id] = {'label': LABELS[label_index], 'confidence': round(label_confidence, 4)}




                # Label prediction through voting on each claim/rationale pair
                # Copied from another file to test it here
                # Does not work as well as concatenating all rationale sentences

                # # vote by mutilple evidence
                # label_index_list = []
                # label_confidence_list = []
                # logits_list = []
                # for i in indices: 
                #     if "abstracts" in selection:
                #         print('abstracts are in selection')
                #         evidence = selection["abstracts"][doc_id][i]
                #     else:
                #         print('abstracts are not in selection')
                #         evidence = corpus[doc_id]['abstract'][i]                    
                #     encoded_dict = encode([evidence], [claim])
                #     model_output = model(**encoded_dict)[0]
                #     label_scores = torch.softmax(model_output, dim=1)[0]

                #     # Check correlation between claim and evidence 
                #     if args.specificity_thres:
                #         correlation_score = get_specificity(claim, evidence, args.include_nouns, args.fuzzy, 'continuous')
                #         if correlation_score > args.specificity_thres:
                #             label_index_list.append(label_scores.argmax().item())
                #             label_confidence_list.append(label_scores[label_index_list[-1]].item())
                #         else:
                #             label_index_list.append(1)
                #             label_confidence_list.append(1)
                #     else:
                #         label_index_list.append(label_scores.argmax().item())
                #         label_confidence_list.append(label_scores[label_index_list[-1]].item())

                #     logits_list.append(model_output)
                    
                # # voting
                # label_index = max(label_index_list, key=label_index_list.count) 
                # label_confidence = label_confidence_list[label_index_list.index(label_index)]
                
                # logits = logits_list[label_index_list.index(label_index)]
                # sup_refu_indices = torch.tensor([0, 2]).cuda() # index of support and refute
                # probs = torch.softmax(torch.index_select(logits, 1, sup_refu_indices), dim=1)[0]

                # results[doc_id] = {'label': LABELS[label_index], 'confidence': round(label_confidence, 4), "probs": probs.tolist(),
                #     "rational_preds": [{"pred":LABELS[label_index_list[i]], "confidence":round(label_confidence_list[i], 4), "evidence": corpus[doc_id]['abstract'][idx]} for i, idx in enumerate(indices)  ]
                #     }

        output.write({
            'claim_id': data['id'],
            'labels': results
        })