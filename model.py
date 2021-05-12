import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import utils
from transformers import RobertaConfig
from dataclasses import dataclass

@dataclass
class ModelConfig(RobertaConfig):
    context_aware_qa: bool = True
    mask_context_embedding: bool = False
    teacher_forcing: bool = True
    oracle: bool = False
    num_chains: int = 4
    dev_num_chains: int = 5


class BertSequentialReasoningSingleEncoding(nn.Module):
    def __init__(self, config):
        super(BertSequentialReasoningSingleEncoding, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained('roberta-large')
        self.dropout = nn.Dropout()

        self.sentence_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size // 2), nn.ReLU(),
                                                 nn.Linear(config.hidden_size // 2, 1))

        self.qa_output = nn.Linear(
            2 * config.hidden_size if config.context_aware_qa or config.mask_context_embedding else config.hidden_size,
            2)
        self.eoc_vector = nn.Parameter(torch.zeros(1, config.hidden_size))
        nn.init.xavier_uniform_(self.eoc_vector)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    def _encode_context(self, features, sentence_indicator):
        sentences = []
        sentence_lens = []
        output = self.bert(**features)
        question_embedding = None
        question_len = None
        for i in range(sentence_indicator[0][-1]):
            mask = (sentence_indicator == i).long().cuda()
            sentence_embedding = torch.sum(output[0] * mask.unsqueeze(-1), dim=1)
            sentence_len = mask.sum(dim=1).view(-1, 1)
            if i == 0:
                question_embedding = sentence_embedding
                question_len = sentence_len
            else:
                sentences.append(sentence_embedding)
                sentence_lens.append(sentence_len)

        eoc_vector = self.eoc_vector.expand(features['input_ids'].size(0), -1)
        sentences.append(eoc_vector)
        sentence_lens.append(torch.ones(features['input_ids'].size(0), 1, dtype=torch.long).cuda())
        sentences = torch.stack(sentences, dim=1)
        sentence_lens = torch.stack(sentence_lens, dim=1)

        return output[0], question_embedding, question_len, sentences, sentence_lens

    def _sentence_step(self, cur_embedding, cur_emb_len, sentences, sentence_lens, selected_sentence_mask):
        combined_sentence_embeddings = cur_embedding.unsqueeze(1) + sentences
        combined_sentence_len = cur_emb_len.unsqueeze(1) + sentence_lens

        pooled_embeddings = combined_sentence_embeddings / combined_sentence_len
        sentence_logits = self.sentence_classifier(pooled_embeddings).squeeze(-1)
        sentence_logits = utils.mask_tensor(sentence_logits, selected_sentence_mask.detach())
        return sentence_logits, combined_sentence_embeddings, combined_sentence_len

    def _sentence_selection(self, cur_embedding, cur_emb_len, sentences, sentence_lens, selected_sentence_mask,
                            fact_label=None):
        combined_sentence_embeddings = cur_embedding.unsqueeze(1) + sentences
        combined_sentence_len = cur_emb_len.unsqueeze(1) + sentence_lens
        pooled_embeddings = combined_sentence_embeddings / combined_sentence_len
        sentence_logits = self.sentence_classifier(pooled_embeddings).squeeze(-1)
        sentence_logits = utils.mask_tensor(sentence_logits, selected_sentence_mask.detach())

        num_sentences = combined_sentence_embeddings.size(1)

        if self.training:
            if self.config.teacher_forcing and fact_label is not None:
                one_hot = utils.convert_to_one_hot(fact_label, num_sentences)
            else:
                one_hot = F.gumbel_softmax(sentence_logits, tau=0.8, hard=True)
        else:
            if self.config.oracle and fact_label is not None:
                one_hot = utils.convert_to_one_hot(fact_label, num_sentences)
            else:
                one_hot = torch.argmax(sentence_logits, dim=-1)
                one_hot = utils.convert_to_one_hot(one_hot, num_sentences)

        if one_hot[0][-1] != 1:  # ONLY FOR BATCH SIZE 1
            selected_sentence_mask = (1 - one_hot) * selected_sentence_mask

        one_hot = one_hot.unsqueeze(-1)

        new_embedding = (one_hot * combined_sentence_embeddings).sum(dim=1)
        new_embedding_len = (one_hot * combined_sentence_len).sum(dim=1)

        return sentence_logits, new_embedding, new_embedding_len, selected_sentence_mask, one_hot.squeeze(-1)

    def forward(
            self,
            features,
            sentence_indicator,
            fact_label=None
    ):
        context_embedding, question_embedding_sum, question_len, sentences_sum, sentence_lens = self._encode_context(
            features, sentence_indicator)
        batch_size, num_sentences = sentences_sum.size(0), sentences_sum.size(1)
        selected_sentence_mask = torch.ones(batch_size, num_sentences).cuda()
        selected_sentences_one_hot = torch.zeros(batch_size, num_sentences).cuda()

        cur_embedding = question_embedding_sum
        cur_emb_len = question_len
        chain_logits = []
        num_chains = self.config.num_chains if self.training else self.config.dev_num_chains
        for i in range(num_chains):
            sentence_logits, cur_embedding, cur_emb_len, selected_sentence_mask, one_hot = self._sentence_selection(
                cur_embedding,
                cur_emb_len,
                sentences_sum,
                sentence_lens,
                selected_sentence_mask,
                fact_label=fact_label[:, i] if fact_label is not None else None
            )

            chain_logits.append(sentence_logits)
            selected_sentences_one_hot = selected_sentences_one_hot + one_hot

            if one_hot[0][-1] == 1:  # ONLY FOR BATCH SIZE 1
                break

        selected_sentences_one_hot = selected_sentences_one_hot.clamp(max=1)
        attention_mask = utils.convert(sentence_indicator, selected_sentences_one_hot).type(torch.long)

        input_ids = features['input_ids'] * (attention_mask) + (1 - attention_mask) * self.tokenizer.pad_token_id

        outputs = self.bert(input_ids.long(), attention_mask=attention_mask)
        sequence_output = outputs[0]

        if self.config.mask_context_embedding:
            context_embedding = attention_mask * context_embedding

        if self.config.context_aware_qa or self.config.mask_context_embedding:
            sequence_output = torch.cat((sequence_output, context_embedding), dim=-1)

        logits = self.qa_output(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1), chain_logits