import argparse
import jsonlines
import os

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=False)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--label-prediction', type=str, required=True)
parser.add_argument('--filter', type=str, choices=['structured', 'unstructured'])
parser.add_argument('--deleting-model-path', type=str, default=None, required=False)

args = parser.parse_args()

#corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
label_prediction = jsonlines.open(args.label_prediction)

pred_labels = []
true_labels = []

LABELS = {'REFUTES': 0, 'NOT ENOUGH INFO': 1, 'SUPPORTS': 2}

for data, prediction in zip(dataset, label_prediction):
    assert data['id'] == prediction['claim_id']

    if args.filter:
        prediction['labels'] = {doc_id: pred for doc_id, pred in prediction['labels'].items()
                                if corpus[int(doc_id)]['structured'] is (args.filter == 'structured')}
    if not prediction['labels']:
        continue

    claim_id = data['id']
    for doc_id, pred in prediction['labels'].items():
        pred_label = pred['label']
        # true_label = {es['label'] for es in data['evidence'].get(doc_id) or []}
        true_label = {data["label"]}
        assert len(true_label) <= 1, 'Currently support only one label per doc'
        true_label = next(iter(true_label)) if true_label else 'NOT ENOUGH INFO'
        pred_labels.append(LABELS[pred_label])
        true_labels.append(LABELS[true_label])

print(f'Accuracy           {round(sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels), 4)}')
print(f'Macro F1:          {f1_score(true_labels, pred_labels, average="macro").round(4)}')
print(f'Macro F1 w/o NEI:  {f1_score(true_labels, pred_labels, average="macro", labels=[0, 2]).round(4)}')
print()
print('                   [C      N      S     ]')
print(f'F1:                {f1_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Precision:         {precision_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Recall:            {recall_score(true_labels, pred_labels, average=None).round(4)}')
print()
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))

#remove the evaluated model for space saving
if args.deleting_model_path and sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels) < 0.72 and f1_score(true_labels, pred_labels, average="macro") < 0.70:
    if os.path.exists(args.deleting_model_path):
        for root, subdirs, files in os.walk(args.deleting_model_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(args.deleting_model_path)
