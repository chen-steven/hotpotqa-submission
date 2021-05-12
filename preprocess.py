import json
from tqdm import tqdm
import pickle
from collections import defaultdict
from transformers import RobertaTokenizerFast


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_all_one_hot(tokens, text, ans, i=None):
    ids = tokens['input_ids'][i] if i is not None else tokens['input_ids']
    start_one_hot, end_one_hot = [0] * len(ids), [0] * len(ids)

    all_idxs = list(find_all(text, ans))
    for idx in all_idxs:
        if i is None:
            s_label = tokens.char_to_token(idx)
            e_label = tokens.char_to_token(idx + len(ans) - 1)
        else:
            s_label = tokens.char_to_token(i, idx)
            e_label = tokens.char_to_token(i, idx + len(ans) - 1)

        try:
            start_one_hot[s_label] = 1
            end_one_hot[e_label] = 1
        except Exception:
            #            print(s_label, e_label)
            pass
    # print(text, ans, s_label, e_label)
    return start_one_hot, end_one_hot


def get_examples(file_path, paras_file):
    paras = json.load(open(paras_file, 'r'))
    sentence2title = defaultdict(list)
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for article in tqdm(data):
            uid = article["id"]
            question = "yes no. " + article['question']
            context = article["context"]
            assert len(context) >= 2
            context_sents, fact_labels = [], []

            selected_paras = []
            for title, sents in context:
                if title in paras[uid]:
                    context_sents.extend(sents)
                    selected_paras.append((title, sents))

            for title, sents in selected_paras:
                for i in range(len(sents)):
                    sentence2title[uid].append((title, i))

            example = {"id": uid, "question": question,
                       "context_sent": context_sents}
            examples.append(example)

    json.dump(sentence2title, open(f'data/sentence2para.json', 'w'))
    json.dump(example, open(f'data/examples.json', 'w'))
    return examples


def build_hotpot_single_encoding_features(examples, filename):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    ids = []
    full_context = []
    sentence_indicators = []
    for example in examples:
        ids.append(example['id'])
        context_sentences = example['context_sent']
        question = example['question']

        input_text = question
        for i, sent in enumerate(context_sentences):
            input_text += ' {} {}'.format(tokenizer.sep_token, sent)

        full_context.append(input_text)

    features = tokenizer(full_context, truncation=True)
    for i, example in tqdm(enumerate(examples)):

        sentence_indicator = [0] * len(features['input_ids'][i])
        sent_idx = 0

        for idx, x in enumerate(features['input_ids'][i]):
            if x == tokenizer.sep_token_id:
                sent_idx += 1

            sentence_indicator[idx] = sent_idx
        sentence_indicator[-1] = 0
        sentence_indicators.append(sentence_indicator)

    data = dict(features=features,
                sentence_indicators=sentence_indicators,
                ids=ids)

    pickle.dump(data, open(filename, 'wb'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str)
    args = parser.parse_args()
    examples = get_examples(args.dataset_file, "data/paras.json")
    build_hotpot_single_encoding_features(examples, "data/features.pkl")