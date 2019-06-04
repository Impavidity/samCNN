import torch
import torch.nn as nn
from basic import SimpleCNN
import torch.nn.functional as F

class SM(nn.Module):
    def __init__(self, config):
        super(SM, self).__init__()
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
            nn.Linear(config.hidden_size * 2,
                      config.hidden_layer_units),
            nn.BatchNorm1d(config.hidden_layer_units),
            nn.Tanh(),
            nn.Dropout(config.dropout),
        )

        self.softmax = nn.Sequential(
            nn.Linear(config.hidden_layer_units + ext_feats_size, config.num_classes),
            nn.LogSoftmax()
        )




    def forward(self, question, answer, ext_feats):
        sent1 = self.embedding(question)
        sent2 = self.embedding(answer)
        feature1 = self.sm_cnn(sent1)
        feature2 = self.sm_cnn(sent2)
        feat_comb = torch.cat([feature1, feature2], dim=1)
        feat = self.final_layers(feat_comb)
        if self.config.ext_feats:
            feat = torch.cat([feat, ext_feats], dim=1)
        preds = self.softmax(feat)

        return (preds, feat)