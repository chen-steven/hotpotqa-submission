import torch
import random
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

__DEVICE__ = None


def set_device(device):
    global __DEVICE__
    __DEVICE__ = device


def discretize(p_start, p_end, max_len=15, no_answer=False):
    """Discretize soft predictions to get start and end indices.
    Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
    subject to `i <= j` and `j - i + 1 <= max_len`.
    Args:
        p_start (torch.Tensor): Soft predictions for start index.
            Shape (batch_size, context_len).
        p_end (torch.Tensor): Soft predictions for end index.
            Shape (batch_size, context_len).
        max_len (int): Maximum length of the discretized prediction.
            I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
        no_answer (bool): Treat 0-index as the no-answer prediction. Consider
            a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
            than the probability assigned to the max-probability span.
    Returns:
        start_idxs (torch.Tensor): Hard predictions for start index.
            Shape (batch_size,)
        end_idxs (torch.Tensor): Hard predictions for end index.
            Shape (batch_size,)
    """
    if p_start.min() < 0 or p_start.max() > 1 \
            or p_end.min() < 0 or p_end.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probabilities
    p_start = p_start.unsqueeze(dim=2)
    p_end = p_end.unsqueeze(dim=1)
    p_joint = torch.matmul(p_start, p_end)  # (batch_size, c_len, c_len)

    # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
    c_len, device = p_start.size(1), p_start.device
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
                                diagonal=max_len)
    if no_answer:
        # Index 0 is no-answer
        p_no_answer = p_joint[:, 0, 0].clone()
        is_legal_pair[0, :] = 0
        is_legal_pair[:, 0] = 0
    else:
        p_no_answer = None
    p_joint *= is_legal_pair

    # Take pair (i, j) that maximizes p_joint
    max_in_row, _ = torch.max(p_joint, dim=2)
    max_in_col, _ = torch.max(p_joint, dim=1)
    start_idxs = torch.argmax(max_in_row, dim=-1)
    end_idxs = torch.argmax(max_in_col, dim=-1)

    if no_answer:
        # Predict no-answer whenever p_no_answer > max_prob
        max_prob, _ = torch.max(max_in_col, dim=-1)
        start_idxs[p_no_answer > max_prob] = 0
        end_idxs[p_no_answer > max_prob] = 0

    return start_idxs, end_idxs


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_one_hot(tensor, size):
    one_hot = torch.zeros(*tensor.size(), size).to(__DEVICE__)
    one_hot = one_hot.reshape(-1, one_hot.size(-1))
    pos = tensor.reshape(-1, 1)
    one_hot = one_hot.scatter(1, pos, 1).reshape(*tensor.size(), size)
    return one_hot


def stable_max(tensor):
    best, idx = tensor[:, 0], torch.zeros(tensor.size(0)).type(torch.long).to(__DEVICE__)

    for i in range(tensor.size(0)):
        for j in range(1, tensor.size(1)):
            if tensor[i][j] > best[i]:
                best[i] = tensor[i][j]
                idx[i] = j

    return best, idx


def top_k_with_filter(tensor, k, filtered_element):
    v, i = torch.sort(tensor, descending=True)
    filtered, filtered_i = [], []
    top, top_i = [], []
    for idx in range(len(v)):
        if len(top) == k:
            break
        if i[idx] % (filtered_element + 1) == filtered_element:
            filtered.append(v[idx])
            filtered_i.append(i[idx])
        else:
            top.append(v[idx])
            top_i.append(i[idx])

    return torch.tensor(top, device=tensor.device), torch.tensor(top_i, device=tensor.device), torch.tensor(filtered,
                                                                                                            device=tensor.device), torch.tensor(
        filtered_i, device=tensor.device)


def mask_tensor(tensor, mask, mask_value=-1e30):
    return mask * tensor + (1 - mask) * mask_value


def get_allocated_memory():
    return "{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3)


def convert(sentence_indicator, gumbel_output):
    # add one at the beginning, always keep the question selected
    batch_size, sent_num = gumbel_output.size()
    gumbel_output = torch.cat([torch.ones((batch_size, 1), device=__DEVICE__, dtype=torch.float32), gumbel_output],
                              dim=1)
    batch_idx = torch.range(0, batch_size - 1, device=__DEVICE__, dtype=torch.long).reshape(-1, 1)
    idx = batch_idx * (sent_num + 1) + sentence_indicator
    return gumbel_output.view(-1)[idx]


def convert_attention_mask(indicator, gumbel_output):
    output_we_care = gumbel_output[indicator.bool()]
    sent_idx = torch.cumsum(indicator, dim=1) - 1  # same sentence as the same idx
    sent_num = torch.sum(indicator, dim=1)  # how many sentences
    shift_sent_num = torch.cat([torch.tensor([0], device=__DEVICE__), sent_num[:-1]])
    batch_sent_idx = sent_idx + shift_sent_num.unsqueeze(dim=1)
    return output_we_care[batch_sent_idx]


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


class Processor:
    def __init__(self, args, pref=''):
        self.args = args
        self.pref = pref

    def _get_cached_file_name(self, split):
        res = "{}_{}.pkl".format(self.pref, split)
        return res

    def get_examples(self, filename):
        raise NotImplementedError

    def load_and_cache_examples(self, split='train'):
        if split == 'train':
            examples_file = self.args.train_examples
        elif split == 'dev':
            examples_file = self.args.dev_examples
        else:
            examples_file = self.args.test_examples

        filename = self._get_cached_file_name(split)
        cached_features_file = os.path.join(self.args.data_dir, filename)

        if not os.path.exists(cached_features_file):
            logger.info("Building features and saving to {}".format(cached_features_file))
            examples = self.get_examples(os.path.join(self.args.data_dir, examples_file))
            self.build_features(examples, cached_features_file)
        else:
            logger.info("Found features file in {}".format(cached_features_file))
        return cached_features_file

    def build_features(self, examples, filename):
        raise NotImplementedError


class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, name):
        self.metrics[name] = []

    def get_best(self, name):
        return max(self.metrics[name])

    def update(self, name, val):
        self.metrics[name].append(val)


def gumbel_softmax_topk(logits, k, tau=1, hard=False, dim=-1):
    gumbels = -torch.empty_like(logits,
                                memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(k, dim=dim)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret