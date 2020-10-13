import argparse
import torch
import jsonlines
import os

from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--claim-train', type=str, required=True)
parser.add_argument('--claim-dev', type=str, required=True)
parser.add_argument('--claim-unsup', type=str, required=True)
parser.add_argument('--claim-aug', type=str, required=True)
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-large')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=256, help='The batch size for each gradient update')
parser.add_argument('--batch-size-unsup-ratio', type=float, default=0, help='The batch size to send through GPU')

parser.add_argument('--lr-base', type=float, default=1e-5)
parser.add_argument('--lr-linear', type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument('--no_cuda', type=bool, default=False, help='whether using GPU')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
print(f'Using device "{device}"')

if args.seed:
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
print(f"Training/evaluation parameters {args}")


class SciFactRationaleSelectionDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        self.samples = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        for claim in jsonlines.open(claims):
            for doc_id, evidence in claim['evidence'].items():
                doc = corpus[int(doc_id)]
                evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': claim['claim'],
                        'sentence': sentence,
                        'evidence': i in evidence_sentence_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

trainset = SciFactRationaleSelectionDataset(args.corpus, args.claim_train)
devset = SciFactRationaleSelectionDataset(args.corpus, args.claim_dev)
if args.batch_size_unsup_ratio:
    unsupset = SciFactRationaleSelectionDataset(args.corpus, args.claim_unsup)
    augset = SciFactRationaleSelectionDataset(args.corpus, args.claim_aug)
    assert len(unsupset) == len(augset)
    concatset = ConcatDataset(unsupset, augset)
    batch_size_unsup = int(args.batch_size_gpu * args.batch_size_unsup_ratio)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
optimizer = torch.optim.Adam([
    {'params': model.roberta.parameters(), 'lr': args.lr_base},  # if using non-roberta model, change the base param path.
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)

def encode(claims: List[str], sentences: List[str]):
    encoded_dict = tokenizer.batch_encode_plus(
        zip(sentences, claims),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            zip(sentences, claims),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def evaluate(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['evidence'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return f1_score(targets, outputs, zero_division=0),\
           precision_score(targets, outputs, zero_division=0),\
           recall_score(targets, outputs, zero_division=0)

def unsup_feedforward(model, batch_unsup, batch_aug):
    logSoftmax_fct = torch.nn.LogSoftmax(dim=-1)
    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")

    encoded_dict_unsup = encode(batch_unsup['claim'], batch_unsup['sentence'])
    encoded_dict_aug = encode(batch_aug['claim'], batch_aug['sentence'])
    logits_u = model(**encoded_dict_unsup)
    logits_aug = model(**encoded_dict_aug)
    #logits_u = logSoftmax_fct(logits_u[0])
    logits_aug = logSoftmax_fct(logits_aug[0])
    loss_unsup = loss_fct(logits_aug, logits_u[0].detach())
    #import pdb; pdb.set_trace()

    return loss_unsup

if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

for e in range(args.epochs):

    model.train()
    t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
    if args.batch_size_unsup_ratio:
        concat_loader = DataLoader(concatset, batch_size=batch_size_unsup, shuffle=True)
        concat_repeated = cycle(concat_loader)
    for i, batch in enumerate(t):
        encoded_dict = encode(batch['claim'], batch['sentence'])
        loss, logits = model(**encoded_dict, labels=batch['evidence'].long().to(device))
        if args.batch_size_unsup_ratio:
            batch_unsup, batch_aug = next(concat_repeated)
            loss_unsup = unsup_feedforward(model, batch_unsup, batch_aug)
            loss = loss + loss_unsup
        
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        #if args.gradient_accumulation_steps > 1:
        #    loss = loss / args.gradient_accumulation_steps
        
        loss.backward()
        if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
    scheduler.step()
    train_score = evaluate(model, trainset)
    print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f' % train_score)
    dev_score = evaluate(model, devset)
    print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f' % dev_score)
    # Save
    save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score[0] * 1e4)}')
    os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.module.save_pretrained(save_path)
