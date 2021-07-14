import torch
import torch.nn as nn
from transformers import AutoModel, RobertaModel
import util


class OneStepSentence(nn.Module):
    def __init__(self, config):
        super(OneStepSentence, self).__init__()
        self.config = config
        self.bert = RobertaModel.from_pretrained("data/models/finetuned/PS")
        self.chain_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size // 2),
                                                  nn.LeakyReLU(), nn.Linear(config.hidden_size // 2, 1))
        self.qa_output = nn.Linear(2*config.hidden_size, 2)
        self.pad_id = 1

    def forward(self, features, sentence_indicator, fact_label=None, eval=False):
        output = self.bert(**features)
        context_embedding = output[0]  # full context embedding
        sentences = []
        for i in range(1, sentence_indicator[0][-1]+1):
            mask = (sentence_indicator == i).long().cuda()
            sentences.append(
                torch.sum(output[0] * mask.unsqueeze(-1), dim=1) / (mask.sum(dim=1).view(-1, 1) + 1e-12))
        sentences = torch.stack(sentences, dim=1)
        chain_logits = self.chain_classifier(sentences) if self.config.k == 0 else \
            self.chain_classifier(sentences).squeeze(dim=-1)

        gumbel_output = torch.topk(chain_logits, min(sentence_indicator[0][-2], self.config.k), dim=-1)[1]
        gumbel_output = torch.zeros_like(chain_logits, memory_format=torch.legacy_contiguous_format
                                         ).scatter_(1, gumbel_output, 1.0)

        att_mask = util.convert(sentence_indicator, gumbel_output).long()
        input_ids = features['input_ids'] * att_mask + (1 - att_mask) * self.pad_id
        output = self.bert(input_ids.long(), attention_mask=att_mask)
        sequence_output = output[0]

        context_embedding = att_mask.unsqueeze(dim=-1)*context_embedding

        sequence_output = torch.cat((sequence_output, context_embedding), dim=-1)

        logits = self.qa_output(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        return start_logits, end_logits, chain_logits
