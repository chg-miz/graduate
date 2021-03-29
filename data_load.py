import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS, bert_vocab_path, bert_model_path
from utils import build_vocab
from transformers import BertTokenizer

# init vocab
# ACE2005的34种事件类型转为索引
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
# ACE2005标记的54种实体类型
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
# 斯坦福NLP工具中的词性标签
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
# ACE的35中参数角色转为索引【时间有8种，这里全部视为一种，所以共28种】
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)

# tokenizer = BertTokenizer.from_pretrained('/home/chg/PycharmProjects/EE/model/PretrainModel/bert-base-uncased/bert-base-uncased-vocab.txt',
tokenizer = BertTokenizer.from_pretrained(bert_vocab_path,
                                          do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))


# 输入为ACE2005预处理好了的json文件
class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath):
        # 整个数据集的：句子、实体、标签、触发词、论元
        # senti_li: 整个数据集的句子，每个句子已经加了Bert的标签 [[CLS、XXX、...、SEP]...]
        # entities_li： 整个数据集的实体的词情况， [[[PAD],[B-XXX],[I-XXX]...，[PAD]]...]     最里面的list对应一个词，元素为该词对应实体的标签，只有该词出现在多个实体中时，才会有多个值；第二层的list对应一个句子；
        #                                                              这么看也就是将一个句子的所有词替换成对应实体的标签集合，如果不属于任何实体则用[None]表示，如果对应到多个实体则实体标签集合中放多个值
        # postags_li： 整个数据集的句子， [[PAD、XXX、...、PAD]...]      将一个句子的所有词替换成对应词性的标签
        # triggers_li: 整个数据集的句子， [[XXX、...]...]     将一个句子的所有词替换成对应触发词的标签，如果不属于任何触发词则用None表示，注意这个没有加PAD，是触发词分类的标准结果
        # arguments_li：整个数据集的句子， [{'candidates': [(5, 6, "entity_type_str"),...], 'events': {(1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]...}...]
        #                                                           句子中实体起始、终止、类型信息，句子中触发词对应的角色
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li = [], [], [], [], []
        # ///////////////////////////////////////////////////////////////////////// 上下文句子信息
        self.pre_sent_li = []
        self.next_sent_li = []

        # 加载数据集
        with open(fpath, 'r') as f:
            data = json.load(f)
            # data为列表；item为字典，对应每句话，键有：sentence、golden-entity-mentions、golden-event-mentions、stanford-colcc、words、pos-tags、lemma、parse
            for item in data:
                # words: ["Earlier","documents",...]
                words = item['words']
                # entities： [[NONE],[NONE]...]  长度为words_len
                # entities中每个元素（List）对应一个词，List中每个元素该词对应的实体的类型，只有一个词在多个个实体中时里面的List才会有多个值。[[B-XXX],[I-XXX],...]
                entities = [[NONE] for _ in range(len(words))]
                # triggers： [NONE,NONE...]  长度为words_len
                # 如果该位置的词不是触发词中的词，则保持为NOne；如果是，而且是触发词中第一个词则记为 B-XXX，是触发词中其他词记为 I-XXX   这个就是触发词预测的标准结果
                triggers = [NONE] * len(words)
                # pos-tags: ["JJR","NNS",...]  长度为words_len
                postags = item['pos-tags']
                # arguments: {'candidates': [(5, 6, "entity_type_str"),...], 'events': {(1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...}}
                # candidates 中 (实体开始词索引，结束词索引，实体类型)
                # events 中  (触发词的开始词索引、结束词索引、事件类型): (角色开始词索引、角色结束词索引、角色类型索引)                这个是论元预测的标准结果
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }

                #  golden-entity-mentions
                #  {
                #        {
                #         "text": "Welch",
                #         "entity-type": "PER:Individual",
                #         "head": {
                #           "text": "Welch",
                #           "start": 11,
                #           "end": 12
                #         },
                #         "entity_id": "APW_ENG_20030325.0786-E24-38",
                #         "start": 11,
                #         "end": 12
                #       },
                # ...
                # }
                for entity_mention in item['golden-entity-mentions']:
                    # 实体开始词索引、结束词索引、实体类型
                    arguments['candidates'].append(
                        (entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))
                    # 为实体中的每个词加上实体类型标签，实体第一个词的标签为 B-XXX，其他词为I-XXX
                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)
                        # 将实体的词的标签存到entities中，[[B-XXX],[I-XXX]...]
                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            # 这里有个问题，只有一个词在多个实体中时，才会执行到一步
                            entities[i].append(entity_type)

                #  golden-event-mentions: 一句话里面可以有多个事件提及
                # [
                #     {
                #         "trigger": {
                #           "text": "retirement",
                #           "start": 17,
                #           "end": 18
                #         },
                #         "arguments": [
                #           {
                #             "role": "Person",
                #             "entity-type": "PER:Individual",
                #             "text": "Welch",
                #             "start": 11,
                #             "end": 12
                #           },
                #           {
                #             "role": "Entity",
                #             "entity-type": "ORG:Commercial",
                #             "text": "GE",
                #             "start": 20,
                #             "end": 21
                #           }
                #         ],
                #         "event_type": "Personnel:End-Position"
                #      }
                # ...
                # ]
                # 每个事件提及是分开标记的，应该会有同一个实体是多个事件中角色的情况
                for event_mention in item['golden-event-mentions']:
                    # 触发词中的每个词加上触发词类型标签，触发词第一个词的标签为 B-XXX，其他词为I-XXX
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)
                    # 触发词的开始词索引、结束词索引、事件类型，作为事件关键因素
                    event_key = (
                        event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        # 表示时间的角色有8种，全部作为一种处理
                        if role.startswith('Time'):
                            role = role.split('-')[0]
                        # 将角色类型转为角色类型索引，   角色开始词索引、角色结束词索引、角色类型索引
                        arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))
                # python的append操作,合并list(注意:不是extend合并list的值)
                # 句子首尾加上Bert标签，句子内容变了，其他几个也要跟着变，下面是变化
                self.sent_li.append([CLS] + words + [SEP])
                # [CLS]、[SEP]标签的实体类型、词性标签都是是PAD
                self.entities_li.append([[PAD]] + entities + [[PAD]])
                self.postags_li.append([PAD] + postags + [PAD])
                # 注意： 事件类型类型这个没有加PAD
                self.triggers_li.append(triggers)
                self.arguments_li.append(arguments)
                # ///////////////////////////////////////////////////////////////////////// 上下文句子信息
                pre_sent_words = item['sentence_context_word'][0]
                next_sent_words = item['sentence_context_word'][1]
                self.pre_sent_li.append([CLS] + pre_sent_words + [SEP])
                self.next_sent_li.append([CLS] + next_sent_words + [SEP])

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        # words:        [CLS、XXX、...、SEP]                        长度为句子长度+2
        # entities:     [[PAD],[B-XXX],[I-XXX]...，[PAD]]          长度为句子长度+2
        # postags:      [PAD、XXX、...、PAD]                        长度为句子长度+2
        # triggers:     [、XXX、...、]        事件类型                长度为句子长度      这个很特殊，没有加PAD
        # arguments:    {'candidates': [(5, 6, "entity_type_str"),...], 'events': {(1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]...}
        words, entities, postags, triggers, arguments = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], \
                                                        self.triggers_li[idx], self.arguments_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        # 依次取句子中的每个词；w为词(CLS)，e为词对应的实体标签（[B-XXX]），p为词对应的词性标签(XXX)
        for w, e, p in zip(words, entities, postags):
            # bert分词
            # 将词变为list，CLS变为[CLS]; 就算只有一个词也就是list，何况还有可能出现ing被分开的情况
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            # 从词的List转为Bert中索引的List
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            # 首尾字符用[0]表示、其他字符用[1,0,0...]表示,is_head可以理解为标记分词后的词是不是原词的第一个词(关键词?)
            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            # tag变为 [tag,PAD,...]
            p = [p] + [PAD] * (len(tokens) - 1)
            # [B-XXX]变为 [[B-XXX],[PAD]...]
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            # tag转为索引
            p = [postag2idx[postag] for postag in p]
            # entity转为索引
            e = [[entity2idx[entity] for entity in entities] for entities in e]
            # python的extend操作,合并list的值(注意:不是append合并list)
            # token_x: [CLS,read,ing,...]  把一个句子用bert的分词进行分词,属于一个词的token,用list框起来
            # postags_x: [PAD,V,PAD,...] 与上面对应, 分出来的词的后缀对应的是PAD
            # entities_x: [[PAD],[B-XXX],[PAD],...] 与上面对应, 分出来的词的后缀对应的是[PAD]
            # is_heads:　[0,1,0...]          与上面对应, 分出来的词的后缀对应的是0
            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)
        # 触发词类型（事件类型）转为索引，注意：triggers是长度是分词前的状态
        triggers_y = [trigger2idx[t] for t in triggers]
        # 分词后关键词的索引（非后缀，有实义的词）
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)
        # 句子长度
        seqlen = len(tokens_x)

        # ///////////////////////////////////////////////////////////////////////// 上下文句子信息
        pre_sent_words = self.pre_sent_li[idx]
        next_sent_words = self.next_sent_li[idx]
        pre_sent_tokens_x = []
        next_sent_tokens_x = []
        for w in pre_sent_words:
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
            pre_sent_tokens_x.extend(tokens_xx)
        for w in next_sent_words:
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
            next_sent_tokens_x.extend(tokens_xx)
        pre_sent_len = len(pre_sent_tokens_x)
        next_sent_len = len(next_sent_tokens_x)

        # token_x: [CLS,read,ing,...]  把一个句子用bert的分词进行分词,属于一个词的token,用list框起来
        # postags_x: [PAD,V,PAD,...] 与上面对应, 分出来的词的后缀对应的是PAD
        # entities_x: [[PAD],[B-XXX],[PAD],...] 与上面对应, 分出来的词的后缀对应的是[PAD]
        # is_heads:　[0,1,0...]          与上面对应, 分出来的词的后缀对应的是0
        # triggers_y： [1,2,...]         触发词类型（事件类型）转为索引
        # arguments:    {'candidates': [(5, 6, "entity_type_str"),...], 'events': {(1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]...}
        # seqlen: 10                     分词后句子长度
        # head_indexes: [1,3,4...]       分词后关键词的索引（非后缀，有实义的词）
        # words:    [CLS、XXX、...、SEP]  bert分词前的，原数据集中的分词情况
        # triggers：[XXX、...]   触发词类型（事件类型）【注意这个没有加PAD，长度比上面的小2】
        # return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers
        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, \
               pre_sent_tokens_x, next_sent_tokens_x, pre_sent_len, next_sent_len

    # 训练样本中的句子可能高度不平衡： 包含事件的句子太少，不包含事件的句子太多
    # 在从batch中采样时，要给包含事件的句子更多的权重，这样才能平衡一点
    def get_samples_weight(self):
        samples_weight = []
        # triggers_li: 整个数据集的句子， [[PAD、XXX、...、PAD]...]     将一个句子的所有词替换成对应触发词的标签，如果不属于任何触发词则用None表示，是触发词分类的标准结果
        # triggers: 单个句子， [PAD、XXX、...、PAD]
        # 这里根据句子中是否包含事件，包含事件句子权重为5，不包含为0，得到整个数据集句子的权重情况
        for triggers in self.triggers_li:
            not_none = False
            # trigger是句子中的触发词标签，NONE表示不是触发词
            for trigger in triggers:
                # 注意：triggers中没有加过PAD
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


# dataloader中分batch时进行pad操作
# pad函数作为dataloader的collate_fn参数；
# dataloder返回的是mini_batch的迭代器；根据迭代器取得mini_batch数据后，会调用collate_fn对mini_batch的数据进行处理
def pad(batch):
    # batch为list，里面每一个元素代表一个句子的样本，对应__getitem__返回的一个item
    # (
    # [101, 3041, 2023, 3204, 1051, 1005, 9848, 11866, 6376, 11537, 2058, 2010, 8304, 1997, 4424, 6905, 9989, 2114, 8656, 1012, 102],   长度21
    # [[0], [1], [78], [79], [44], [0], [0],[1], [1], [1], [1], [44, 12], [13], [13],[13], [13], [13], [13], [28, 13], [1], [0]],       长度21
    # [0, 6, 18, 25, 8, 0, 0, 37, 31, 25, 12, 16, 25, 12, 30, 25, 3, 12, 3, 42, 0],                                                     长度21
    # [1, 1, 1, 1, 1, 1, 18, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],                                                                             长度17
    # {'candidates': [(3, 4, 'PER:Individual'), (8, 9, 'PER:Individual'), (15, 16, 'PER:Group'), (8, 16, 'Crime'), (1, 3, 'TIM:time')], 'events': {(6, 7, 'Justice:Trial-Hearing'): [(3, 4, 12), (8, 9, 12), (8, 16, 3), (1, 3, 29)]}},
    # 21,
    # [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],                                                                    长度17
    # ['[CLS]', 'earlier', 'this', 'month', "o'brien", 'narrowly', 'escaped', 'prosecution', 'over', 'his', 'handling', 'of', 'sexual', 'abuse', 'allegations', 'against', 'priests', '.', '[SEP]'],    长度19
    # ['O', 'O', 'O', 'O', 'O', 'O', 'B-Justice:Trial-Hearing', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']                       长度17
    # )
    # zip(*batch)先将batch解包获取多个tuple，然后zip，也就是将tuple中对应位置的元素打包为一个元组，此时结果是zip对象，要转为List
    # 再将上面的元组打包成List，然后再把结果解包赋值给每个变量
    # 原来batch长度为样本个数，现在batch长度为item中字段个数，每个样本的对应字段被合并到一个list中
    # tokens_x_2d： [[101, 3041,...],...]        第一维长度为batch_size，第二维长度为各个句子分词后长度，长度不一
    # entities_x_3d: [[[0], [1],...],...]       第一维长度为batch_size；第二维长度为各个句子分词后长度，长度不一；第三维长度为每个词代表的实体类型的种数，长度不一
    # postags_x_2d： 【[0,6,...], ...]             第一维长度为batch_size；第二维长度为各个句子分词后长度，长度不一
    # triggers_y_2d：[[1,1,...]...]               第一维长度为batch_size；第二维长度为各个句子分词前的长度，而且不包括CLS、SEP，长度不一
    # arguments_2d:  [{...},...]
    # seqlens_1d:   [21,...]                    第一维长度为batch_size；第二维长度为各个句子分词后长度，长度不一
    # head_indexes_2d: [[1, 2, ...],...]        第一维长度为batch_size；第二维长度为各个句子分词前的长度，而且不包括CLS、SEP，长度不一
    # words_2d: [['[CLS]', 'earlier',...],...]  第一维长度为batch_size；第二维长度为各个句子分词前长度，长度不一
    # triggers_2d: [['O', 'O', ...]...]         第一维长度为batch_size；第二维长度为各个句子分词前长度，而且不包括CLS、SEP，长度不一
    # tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = list(
    #     map(list, zip(*batch)))
    tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, \
    pre_sent_tokens_x, next_sent_tokens_x, pre_sent_len, next_sent_len = list(map(list, zip(*batch)))
    # 获取mini_batch中句子的最大长度（分词后的）；这里有个问题啊。如果是用mini_batch中的最大值，那么mini_batch之间item的长度岂不是会不一致？？？ 确实不一样，有影响吗？
    maxlen = np.array(seqlens_1d).max()
    pre_sent_len_max = np.array(pre_sent_len).max()
    next_sent_len_max = np.array(next_sent_len).max()

    # pre_sent_len_max = 0
    # for sent in pre_sent_tokens_x:
    #     pre_sent_len_max = max(pre_sent_len_max, len(sent))
    #
    # next_sent_len_max = 0
    # for sent in next_sent_tokens_x:
    #     next_sent_len_max = max(next_sent_len_max, len(sent))

    maxlen = max(max(pre_sent_len_max, next_sent_len_max), maxlen)

    # 这些都补齐为最长句子长度，有些之前没有包含PAD、没有包含分词的也都要补齐长度
    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        postags_x_2d[i] = postags_x_2d[i] + [0] * (maxlen - len(postags_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]
        pre_sent_tokens_x[i] = pre_sent_tokens_x[i] + [0] * (maxlen - len(pre_sent_tokens_x[i]))
        next_sent_tokens_x[i] = next_sent_tokens_x[i] + [0] * (maxlen - len(next_sent_tokens_x[i]))

    # words_2d、triggers_2d没有补齐
    return tokens_x_2d, entities_x_3d, postags_x_2d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d, \
           pre_sent_tokens_x, next_sent_tokens_x, pre_sent_len, next_sent_len, maxlen
