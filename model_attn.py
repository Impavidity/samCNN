import torch
from torch import nn
from basic import SimpleCNN, MLP
import math
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.word_num, config.word_embed_dim)
        self.embedding.weight.requires_grad = True
        ext_feats_size = 2 if config.ext_feats else 0

        self.sm_cnn = SimpleCNN(
            num_of_conv=1,
            in_channels=1,
            out_channels=config.output_channel,
            kernel_size=[config.kernel_size],
            in_features=config.word_embed_dim,
            out_features=config.hidden_size,
            active_func=nn.ReLU(),
            dropout=config.dropout,
            fc_layer=True
        )


        self.final_layers = nn.Sequential(
            nn.Linear(config.hidden_size * 3,
                      config.hidden_layer_units),
            nn.BatchNorm1d(config.hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(config.dropout)
        )


        self.softmax = nn.Sequential(
            nn.Linear(config.hidden_layer_units + ext_feats_size, config.num_classes),
            nn.LogSoftmax(dim=1)
        )

        self.context_cnn = SimpleCNN(
            num_of_conv=1,
            in_channels=1,
            out_channels=config.output_channel,
            kernel_size=[config.kernel_size],
            in_features=config.word_embed_dim,
            out_features=config.hidden_size,
            active_func=nn.ReLU(),
            dropout=config.dropout,
            fc_layer=True
        )

        self.cnn_attn = MLP(
            in_features=config.hidden_size + (config.word_embed_dim if config.gating_source != "cnn" else config.hidden_size),  # config.lstm_hidden * 2,
            out_features=config.attn_hidden,
            activation=nn.Tanh()
        )
        self.cnn_prob = nn.Linear(
            in_features=config.attn_hidden,
            out_features=1,
            bias=False
        )

        self.attn_softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.key_linear = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.value_linear = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.query_linear = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.tfidf = nn.Embedding(config.word_num, 1)


    def attn_context_ave(self, question_embed, answer_embed, batch_size):
        question_len = question_embed.size(1)
        answer_len = answer_embed.size(1)
        dimension = question_embed.size(2)
        question = torch.cat([question_embed.unsqueeze(2)] * answer_len, dim=2).view(-1, dimension)
        answer = torch.cat([answer_embed.unsqueeze(1)] * question_len, dim=1).view(-1, dimension)
        # (batch, question_len, answer_len, dim)
        attn_prob = self.cos(answer, question).unsqueeze(1)
        attn_answer = (answer * attn_prob).view(batch_size * question_len, answer_len, dimension)
        feature = self.context_cnn(attn_answer).view(batch_size, question_len, -1)
        feature = torch.sum(feature, dim=1) / question_len
        return feature

    def attn_context(self, question_embed, answer_embed, batch_size):
        question_len = question_embed.size(1)
        answer_len = answer_embed.size(1)
        dimension = question_embed.size(2)
        question = torch.cat([question_embed.unsqueeze(2)] * answer_len, dim=2).view(-1, dimension)
        answer = torch.cat([answer_embed.unsqueeze(1)] * question_len, dim=1).view(-1, dimension)
        # (batch, question_len, answer_len, dim)
        attn_prob = self.cos(answer, question).unsqueeze(1)
        attn_answer = (answer * attn_prob).view(batch_size * question_len, answer_len, dimension)
        feature = self.context_cnn(attn_answer)
        # (batch * question_len, feature)
        # question_embed = (batch, question_length, dim)
        qa_comb = torch.cat([feature, question_embed.view(-1, dimension)], dim=1)
        cnn_prob = self.attn_softmax(self.cnn_prob(self.cnn_attn(qa_comb)).view(batch_size, question_len))
        print(cnn_prob)
        # (batch * question_len, feature) * (batch_size * question_len)
        feature = (feature.view(batch_size, question_len, -1) * cnn_prob.unsqueeze(2)).view(batch_size, question_len, -1)
        feature = torch.sum(feature, dim=1)
        return feature

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        # key = (batch_size, dim) value = (batch_size, question_len, dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def self_attn(self, question_embed, answer_embed, question_feature, answer_feature, batch_size):
        # answer_feature = (batch_size, dim)
        question_len = question_embed.size(1)
        answer_len = answer_embed.size(1)
        dimension = question_embed.size(2)
        question = torch.cat([question_embed.unsqueeze(2)] * answer_len, dim=2).view(-1, dimension)
        answer = torch.cat([answer_embed.unsqueeze(1)] * question_len, dim=1).view(-1, dimension)
        # (batch, question_len, answer_len, dim)
        attn_prob = self.cos(answer, question).unsqueeze(1)
        attn_answer = (answer * attn_prob).view(batch_size * question_len, answer_len, dimension)
        feature = self.context_cnn(attn_answer).view(batch_size, question_len, -1)
        # (batch, question_len, feature)
        feature, attn = self.attention(query=self.query_linear(question_feature), key=self.key_linear(feature), value=self.value_linear(feature))
        # (batch, question_len, feature)
        feature = torch.sum(feature, dim=1)
        return feature

    def attn_context_tfidf(self, question_embed, answer_embed, batch_size, query_ids):
        question_len = question_embed.size(1)
        answer_len = answer_embed.size(1)
        dimension = question_embed.size(2)
        question = torch.cat([question_embed.unsqueeze(2)] * answer_len, dim=2).view(-1, dimension)
        answer = torch.cat([answer_embed.unsqueeze(1)] * question_len, dim=1).view(-1, dimension)
        # (batch, question_len, answer_len, dim)
        attn_prob = self.cos(answer, question).unsqueeze(1)
        attn_answer = (answer * attn_prob).view(batch_size * question_len, answer_len, dimension)
        feature = self.context_cnn(attn_answer).view(batch_size, question_len, -1)
        query_weight = self.tfidf(query_ids)
        normalized_weight = F.softmax(query_weight, dim=-2)
        # (batch, question_len, 1)
        feature = torch.sum(feature * normalized_weight, dim=1)
        return feature


    def forward(self, question, answer, ext_feats):
        sent1 = self.embedding(question)
        sent2 = self.embedding(answer)

        feature1 = self.sm_cnn(sent1)
        feature2 = self.sm_cnn(sent2)
        if self.config.gating_source == "embed":
            feature3 = self.attn_context(sent1, sent2, sent1.size(0))
        elif self.config.gating_source == "ave":
            feature3 = self.attn_context_ave(sent1, sent2, sent1.size(0))
        elif self.config.gating_source == "self-attn":
            feature3 = self.self_attn(sent1, sent2, feature1, feature2, sent1.size(0))
        elif self.config.gating_source == "tf-idf":
            feature3 = self.attn_context_tfidf(sent1, sent2, sent1.size(0), question)

        feat_comb = torch.cat([feature1, feature2, feature3], dim=1)
        feat = self.final_layers(feat_comb)
        if self.config.ext_feats:
            feat = torch.cat([feat, ext_feats], dim=1)
        preds = self.softmax(feat)
        return (preds, feat)