from model import BertSequentialReasoningSingleEncoding, ModelConfig
from dataset import OneStepBaselineDataset
from torch.utils.data import DataLoader
import torch
import utils
from transformers import RobertaTokenizerFast
import torch.nn.functional as F
from tqdm import tqdm
import json

def generate_predictions(args):
    sentence2title = json.load(open(args.sentence_title_file, 'r'))
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    model = BertSequentialReasoningSingleEncoding(ModelConfig())
    model.load_state_dict(torch.load('data/model.pt'))
    model.eval()

    dataset = OneStepBaselineDataset(args.features_file)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    all_predictions = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            features, sentence_indicator, ids = batch
            s_logits, e_logits, chain_logits = model(features, sentence_indicator)

            batch_size = s_logits.size(0)

            start_idxs, end_idxs = utils.discretize(F.softmax(s_logits, dim=-1), F.softmax(e_logits, dim=-1))
            for i in range(batch_size):
                start, end = start_idxs[i], end_idxs[i]
                pred_ans = tokenizer.decode(features['input_ids'][i].tolist()[start: end + 1])

                pred_chain = [torch.argmax(x, dim=-1)[i].item() for x in chain_logits]
                pred_supporting_facts = []
                for x in pred_chain:
                    pred_supporting_facts.append(sentence2title[ids[i]][x])
                pred = {'answer': pred_ans, 'sp': pred_supporting_facts}
                all_predictions[ids[i]] = pred

    json.dump(all_predictions, open('predict.json', 'w'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', type=str)

