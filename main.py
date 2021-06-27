from model import BertSequentialReasoningSingleEncoding
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
    config = AutoConfig.from_pretrained('roberta-large')
    config.teacher_forcing = False
    config.oracle = False
    config.num_chains = 4
    config.dev_num_chains = 5
    config.context_aware_qa = True
    config.mask_context_embedding = False
    config.beam_search = False

    sentence2title = json.load(open(args.sentence_map, 'r'))
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    model = BertSequentialReasoningSingleEncoding(config)
    model.load_state_dict(torch.load('model.pt', map_location="cpu"))
    model.eval()
    model.cuda()
    dataset = OneStepBaselineDataset(args.features_file)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    all_predictions = {'answer': {}, 'sp': {}}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            features, sentence_indicator, ids = batch
            s_logits, e_logits, chain_logits = model(features, sentence_indicator)

            batch_size = s_logits.size(0)

            start_idxs, end_idxs = util.discretize(F.softmax(s_logits, dim=-1), F.softmax(e_logits, dim=-1))
            for i in range(batch_size):
                start, end = start_idxs[i], end_idxs[i]
                pred_ans = tokenizer.decode(features['input_ids'][i].tolist()[start: end + 1])

                pred_chain = [torch.argmax(x, dim=-1)[i].item() for x in chain_logits]
                pad_idx = sentence_indicator[0][-2]
                pred_chain = [x for x in pred_chain if x < pad_idx]
                pred_supporting_facts = []
                for x in pred_chain:
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

