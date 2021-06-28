import argparse
import jsonlines

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

#from verisci.inference.label_prediction.specificity import get_specificity

parser = argparse.ArgumentParser()
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

#corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)
output = jsonlines.open(args.output, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).eval().to(device)

LABELS = ['REFUTES', 'NOT ENOUGH INFO', 'SUPPORTS']

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
        assert data['id'] == selection['claim_id']

        claim = data['claim']
        results = {}

        if not selection['evidence']: # deal with non-claim
            results["000"] = {'label': 'NOT ENOUGH INFO', 'confidence': 1}
            
        else:
            for doc_id, indices in selection['evidence'].items():
                if not indices:
                    results[doc_id] = {'label': 'NOT ENOUGH INFO', 'confidence': 1}
                else:
                    # concate all evidences
                    # evidence = ' '.join([data["sentences"][i] for i in indices]) 
                    
                    # only use first evidence
                    # evidence = data["sentences"][indices[0]] 
                    # encoded_dict = encode([evidence], [claim])
                    # label_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[0]
                    # label_index = label_scores.argmax().item()
                    # label_confidence = label_scores[label_index].item()

                    # vote by mutilple evidence
                    label_index_list = []
                    label_confidence_list = []
                    logits_list = []
                    for i in indices: 
                        evidence = data["sentences"][i]
                        encoded_dict = encode([evidence], [claim])
                        model_output = model(**encoded_dict)[0]
                        label_scores = torch.softmax(model_output, dim=1)[0]
                        
                        # Check correlation between claim and evidence 
                        if args.specificity_thres:
                            correlation_score = get_specificity(claim, evidence, args.include_nouns, args.fuzzy, 'continuous')
                            if correlation_score > args.specificity_thres:
                                label_index_list.append(label_scores.argmax().item())
                                label_confidence_list.append(label_scores[label_index_list[-1]].item())
                            else:
                                label_index_list.append(1)
                                label_confidence_list.append(1)
                        else:
                            label_index_list.append(label_scores.argmax().item())
                            label_confidence_list.append(label_scores[label_index_list[-1]].item())

                        logits_list.append(model_output)
                       
                    # voting
                    label_index = max(label_index_list, key=label_index_list.count) 
                    label_confidence = label_confidence_list[label_index_list.index(label_index)]
                    
                    logits = logits_list[label_index_list.index(label_index)]
                    sup_refu_indices = torch.tensor([0, 2]).cuda() # index of support and refute
                    probs = torch.softmax(torch.index_select(logits, 1, sup_refu_indices), dim=1)[0]

                    results[doc_id] = {'label': LABELS[label_index], 'confidence': round(label_confidence, 4), "probs": probs.tolist(),
                        "rational_preds": [{"pred":LABELS[label_index_list[i]], "confidence":round(label_confidence_list[i], 4), "evidence": data["sentences"][idx]} for i, idx in enumerate(indices)  ]
                        }

        output.write({
            'claim_id': data['id'],
            'labels': results
        })
