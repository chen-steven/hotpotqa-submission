from model_ts import OneStepSentence
from dataset import OneStepBaselineDataset
from torch.utils.data import DataLoader
import torch
import util
from transformers import RobertaTokenizerFast, AutoConfig
import torch.nn.functional as F
from tqdm import tqdm
import json

util.set_device(0)


def generate_predictions(args):
    config = AutoConfig.from_pretrained('roberta-large', cache_dir="transformers_cache")
    config.k = 3

    sentence2title = json.load(open(args.sentence_map, 'r'))
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', cache_dir="transformers_cache")
    model = OneStepSentence(config)
    model.load_state_dict(torch.load('model_ts.pt', map_location="cpu"))
    model.eval()
    model.cuda()
    dataset = OneStepBaselineDataset(args.features_file)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    all_predictions = {'answer': {}, 'sp': {}}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            features, sentence_indicators, ids = batch
            s_logits, e_logits, chain_logits = model(features, sentence_indicators)
            s_logits = s_logits.squeeze(-1)
            e_logits = e_logits.squeeze(-1)

            batch_size = chain_logits.size(0)
            start_idxs, end_idxs = util.discretize(F.softmax(s_logits, dim=-1), F.softmax(e_logits, dim=-1))
            for i in range(batch_size):
                start, end = start_idxs[i], end_idxs[i]
                pred_ans = tokenizer.decode(features['input_ids'][i].tolist()[start: end + 1],
                                            skip_special_tokens=True)
                pred_chain = list(chain_logits[i].topk(min(sentence_indicators[i][-2],
                                                           5))[1].detach().cpu().numpy())
                pred_supporting_facts = []
                for x in pred_chain:
                    if x < len(sentence2title[ids[i]]):
                        pred_supporting_facts.append(sentence2title[ids[i]][x])
                all_predictions['answer'][ids[i]] = pred_ans
                all_predictions['sp'][ids[i]] = pred_supporting_facts

    json.dump(all_predictions, open('pred.json', 'w'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', type=str)
    parser.add_argument('--sentence-map', type=str)
    args = parser.parse_args()
    generate_predictions(args)
