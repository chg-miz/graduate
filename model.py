import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel
# from transformers import BertModel
from data_load import idx2trigger, argument2idx
from consts import NONE
from utils import find_triggers

from consts import bert_vocab_path, bert_model_path


class Net(nn.Module):
    def __init__(self, trigger_size=None, entity_size=None, all_postags=None, postag_embedding_dim=50,
                 argument_size=None, entity_embedding_dim=50, device=torch.device("cpu")):
        super().__init__()
        # self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained(bert_model_path)

        # hidden_size = 768 + entity_embedding_dim + postag_embedding_dim
        # hidden_size = 768
        hidden_size = 768 * 3
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        )
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
        )
        self.device = device

    # tokens_x_2d： [[101, 3041,...],...]        第一维长度为batch_size，第二维长度为当前batch_size中句子分词后最大长度
    # entities_x_3d: [[[0], [1],...],...]       第一维长度为batch_size；第二维长度为当前batch_size中句子分词后最大长度；第三维长度为每个词代表的实体类型的种数，长度不一
    # postags_x_2d： 【[0,6,...], ...]             第一维长度为batch_size；第二维长度为当前batch_size中句子分词后最大长度
    # triggers_y_2d：[[1,1,...]...]               第一维长度为batch_size；第二维长度为当前batch_size中句子分词后最大长度
    # arguments_2d:  [{...},...]
    # seqlens_1d:   [21,...]                    第二维长度为当前batch_size中句子分词后最大长度
    # head_indexes_2d: [[1, 2, ...],...]        第二维长度为当前batch_size中句子分词后最大长度
    # words_2d: [['[CLS]', 'earlier',...],...]  第一维长度为batch_size；第二维长度为各个句子分词前长度，长度不一
    # triggers_2d: [['O', 'O', ...]...]         第一维长度为batch_size；第二维长度为各个句子分词前长度，而且不包括CLS、SEP，长度不一

    # 这个模型的batch和PLMEE类似，不是每个词处理后作为样本，而是一整个句子直接输入Bert了，不再用位置嵌入那些东西了
    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d,
                         pre_sent_tokens_x, next_sent_tokens_x,
                         pre_sent_flags,next_sent_flags,pre_sent_len_mat, next_sent_len_mat):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        # postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        # postags_x_2d = self.postag_embed(postags_x_2d)
        # entity_x_2d = self.entity_embed(entities_x_3d)

        if self.training:
            # bert不训练
            self.bert.train()
            # 这个bert没用好吧，没有设置mask
            # 注意了：bert的输出与output_all_encoded_layers参数有关；该值默认为True；
            # 如果output_all_encoded_layers为True，第一个输出为所有层的所有Token的输出，第二个输出是CLS经过全连接层后的输出
            # 如果output_all_encoded_layers为False，第一个输出为最后一层的所有Token的输出，第二个输出是CLS经过全连接层后的输出
            encoded_layers, _ = self.bert(tokens_x_2d)
            # print("encoded_layers.shape" + "="*100)
            # print(len(encoded_layers))
            # print(len(encoded_layers[0]))
            # 这里是取最后一层的输出，其实直接在bert里面设置output_all_encoded_layers为True，就不用手动写这一步了
            enc = encoded_layers[-1]
            # with torch.no_grad():
            #     encoded_layers, _ = self.bert(tokens_x_2d)
            #     enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(tokens_x_2d)
                enc = encoded_layers[-1]

        # x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        x = enc
        # logits = self.fc2(x + enc)

        batch_size = tokens_x_2d.shape[0]

        # 筛选出实义词对应的bert输出，拼接成新的输出x
        # 不是不是！原作者这里也是写瞎了，index_select中的索引是可以重复的；用head_indexes_2d[i]表示索引，而head_indexes_2d[i]的长度本来就是seq_len
        # 导致最后得到的新张量还是seq_len * hidden_size ；而且由于head_indexes_2d[i]末尾是补0的，导致新张量中重复大量x[i][0]，但是话说回来
        # 如果新得到的张量的shape变了，不为seq_len * hidden_size了，反倒是不能重新赋值给x[0]，因为shape不匹配
        # x:[batch_size, seq_len, hidden_size]
        # print("predict_triggers x:" + "="*100)
        # print(head_indexes_2d[0])
        # print(x[0])
        # print(f"{len(x)} + {len(x[0])} + {len(x[0][0])}")
        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])
        # print(f"{len(x)} + {len(x[0])} + {len(x[0][0])}")

        pre_sent_tokens_x = torch.LongTensor(pre_sent_tokens_x).to(self.device)
        next_sent_tokens_x = torch.LongTensor(next_sent_tokens_x).to(self.device)
        pre_sent_len = torch.LongTensor(pre_sent_len_mat).to(self.device)
        next_sent_len = torch.LongTensor(next_sent_len_mat).to(self.device)
        pre_sent_flags = torch.LongTensor(pre_sent_flags).to(self.device)
        next_sent_flags =  torch.LongTensor(next_sent_flags).to(self.device)
        # 实验三：取非[PAD]部分的平均值
        with torch.no_grad():
            pre_sent_encoded_layers, _ = self.bert(pre_sent_tokens_x)
            next_sent_encoded_layers, _ = self.bert(next_sent_tokens_x)
            pre_sent_enc = pre_sent_encoded_layers[-2]
            next_sent_enc = next_sent_encoded_layers[-2]
            # print(pre_sent_enc.shape)
            # pre_sent_tokens_x = torch.mean(pre_sent_enc, 1)
            # print(pre_sent_tokens_x.shape)
            # next_sent_tokens_x = torch.mean(next_sent_enc, 1)
            pre_sent_enc = pre_sent_enc * pre_sent_flags
            next_sent_enc = next_sent_enc * next_sent_flags
            pre_sent_sum = pre_sent_enc.sum(1)
            next_sent_sum = next_sent_enc.sum(1)
            pre_sent_mean = pre_sent_sum / pre_sent_len
            next_sent_mean = next_sent_sum / next_sent_len
            sent_context = torch.cat([pre_sent_mean, next_sent_mean], 1)
            # print(sent_context.shape)
            sent_context = sent_context.unsqueeze(1)
            # print(sent_context.shape)
            sent_context = sent_context.repeat(1, tokens_x_2d.shape[1], 1)
        # print(sent_context.shape)
        x = torch.cat([x, sent_context], 2)

        # 从 batch_size * sentence_len * 768 变为 batch_size * sentence_len * 分类数
        trigger_logits = self.fc_trigger(x)

        # batch_size * sentence_len * 分类数 变为 batch_size * sentence_len
        # print("trigger_hat_2d " + "="*100)
        # print(trigger_logits.shape)
        # argmax 返回最大值所在下标，也即得到了预测分类结果
        trigger_hat_2d = trigger_logits.argmax(-1)
        # print(trigger_hat_2d.shape)

        return trigger_logits, triggers_y_2d, trigger_hat_2d
