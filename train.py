import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from model import Net

from data_load import ACE2005Dataset, pad, all_triggers, all_entities, all_postags, all_arguments, tokenizer
# from utils import report_to_telegram
from eval import eval


def train(model, iterator, optimizer, criterio):
    model.train()
    # batch_size 24 step 600
    for i, batch in enumerate(iterator):
        #
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, \
        pre_sent_tokens_x, next_sent_tokens_x, pre_sent_len, next_sent_len, maxlen = batch

        # maxlen = max(seqlens_1d)
        # pre_sent_len_max = max(pre_sent_len)
        # next_sent_len_max = max(next_sent_len)

        pre_sent_flags = []
        next_sent_flags = []

        pre_sent_len_mat = []
        next_sent_len_mat = []

        for i in pre_sent_len:
            tmp = [[1] * 768] * i + [[0] * 768] * (maxlen - i)
            pre_sent_flags.append(tmp)
            pre_sent_len_mat.append([i] * 768)

        for i in next_sent_len:
            tmp = [[1] * 768] * i + [[0] * 768] * (maxlen - i)
            next_sent_flags.append(tmp)
            next_sent_len_mat.append([i] * 768)

        optimizer.zero_grad()
        # trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
        trigger_logits, triggers_y_2d, trigger_hat_2d = model.predict_triggers(tokens_x_2d=tokens_x_2d,
                                                                               entities_x_3d=entities_x_3d,
                                                                               postags_x_2d=postags_x_2d,
                                                                               head_indexes_2d=head_indexes_2d,
                                                                               triggers_y_2d=triggers_y_2d,
                                                                               arguments_2d=arguments_2d,
                                                                               pre_sent_tokens_x=pre_sent_tokens_x,
                                                                               next_sent_tokens_x=next_sent_tokens_x,
                                                                               pre_sent_flags=pre_sent_flags,
                                                                               next_sent_flags=next_sent_flags,
                                                                               pre_sent_len_mat=pre_sent_len_mat,
                                                                               next_sent_len_mat=next_sent_len_mat)
        # print("trigger_logits 1 " + "="*100)
        # print(trigger_logits.shape)
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])

        print("trigger_check " + "=" * 100)
        print(torch.argmax(trigger_logits, 1).to('cpu').numpy().tolist())
        print(triggers_y_2d.view(1, -1).to('cpu').numpy().tolist())
        # print("trigger_logits 2 " + "="*100)
        # print(trigger_logits.shape)
        # print("triggers_y_2d 1 " + "="*100)
        # print(triggers_y_2d.shape)
        # print("triggers_y_2d 2 " + "=" * 100)
        # print(triggers_y_2d.view(-1).shape)
        # ?????????????????????????????????????????????model???????????????trigger_logits??? batch_size * sentence_len * ????????? ?????????????????? batch_size * sentence_len??? ??????????????????????????????????????????
        # ??????????????? batch_size???sentence???????????????????????????????????????????????????????????????????????????batch?????????????????????
        # CrossEntropyLoss???????????? ???????????????2????????? batch_size * ?????????    ???????????????????????????  batch_size ; ???????????????????????????batch_size ?????? batch_size * sentence_len
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))
        # print("triggers_loss 1 " + "=" * 100)
        # print(trigger_loss.shape)

        loss = trigger_loss

        # ????????????
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()

        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("tokens_x_2d[0]:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
            print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
            print("postags_x_2d[0]:", postags_x_2d[0][:seqlens_1d[0]])
            print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
            print("triggers_2d[0]:", triggers_2d[0])
            print("triggers_y_2d[0]:", triggers_y_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
            print('trigger_hat_2d[0]:', trigger_hat_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
            print("seqlens_1d[0]:", seqlens_1d[0])
            print("arguments_2d[0]:", arguments_2d[0])
            print("=======================")

        if i % 10 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


if __name__ == "__main__":
    # batch_size ??? 24 ?????????11g??????
    parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logdir")
    # parser.add_argument("--trainset", type=str, default="data/train.json")
    # parser.add_argument("--devset", type=str, default="data/dev.json")
    # parser.add_argument("--testset", type=str, default="data/test.json")
    parser.add_argument("--trainset", type=str, default="data/train_chg.json")
    parser.add_argument("--devset", type=str, default="data/dev_chg.json")
    parser.add_argument("--testset", type=str, default="data/test_chg.json")

    parser.add_argument("--telegram_bot_token", type=str, default="")
    parser.add_argument("--telegram_chat_id", type=str, default="")

    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ????????????
    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        entity_size=len(all_entities),
        all_postags=len(all_postags),
        argument_size=len(all_arguments)
    )
    epoch_rec = 0
    # ??????????????????
    # model = torch.load("./logdir/./logdir/checkpoint-7.pkl")
    # ??????????????????
    # checkpoint = torch.load("./logdir/checkpoint-5.pkl")
    # model.load_state_dict(checkpoint['model'])
    # epoch_rec = checkpoint['epoch']
    # print("+"*50)
    # print(epoch_rec)

    # ????????????GPU
    # if device == 'cuda':
    #     model = model.cuda()
    model.to(device)
    # ??????GPU
    # model = nn.DataParallel(model)

    writer = SummaryWriter()
    # ?????????dataset??????ACE2005???????????????????????????json??????
    train_dataset = ACE2005Dataset(hp.trainset)
    dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset)
    # ???????????????????????????????????????????????? ????????????????????????????????????????????????????????????,??????batch????????????????????????????????????????????????????????????????????????????????????
    samples_weight = train_dataset.get_samples_weight()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    # dataloder????????????mini_batch????????????????????????????????????mini_batch?????????????????????collate_fn???mini_batch?????????????????????
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 sampler=sampler,
                                 num_workers=4,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    # ?????????????????????
    # ?????????0????????????????????????????????????????????????0??????PAD
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    best_f1 = 0.
    # for epoch in range(0, hp.n_epochs + 1):
    for epoch in range(epoch_rec + 1, hp.n_epochs + 1):
        print(epoch)
        train(model, test_iter, optimizer, criterion)

        fname = os.path.join(hp.logdir, str(epoch))
        print(f"=========eval dev at epoch={epoch}=========")
        metric_dev = eval(model, dev_iter, fname + '_dev')

        print(f"=========eval test at epoch={epoch}=========")
        metric_test = eval(model, test_iter, fname + '_test')

        # ???????????????
        # if hp.telegram_bot_token:
        #     report_to_telegram('[epoch {}] dev\n{}'.format(epoch, metric_dev), hp.telegram_bot_token, hp.telegram_chat_id)
        #     report_to_telegram('[epoch {}] test\n{}'.format(epoch, metric_test), hp.telegram_bot_token, hp.telegram_chat_id)

        # ??????epoch??????????????????????????????????????????
        for name, value in metric_test.items():
            precison, recall, f1 = value
            writer.add_scalar(f'log/{name} precsion', precison, epoch)
            writer.add_scalar(f'log/{name} recall', recall, epoch)
            writer.add_scalar(f'log/{name} f1', f1, epoch)
        # # ??????????????????
        # # torch.save(model, "latest_model.pt")
        # # ??????????????????
        #
        # if epoch % 5 == 0:
        #     path_name = "./logdir/checkpoint-" + str(epoch) + ".pkl"
        #     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #     torch.save(state, path_name)
        # # print(f"weights were saved to {fname}.pt")

        # metric_test ?????????????????? ??????/??????????????????????????? p???r???f1
        classification_f1 = metric_test['trigger classification'][2]
        if classification_f1 > best_f1:
            path_name = f"./logdir/checkpoint.pkl"
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, path_name)
            best_f1 = classification_f1
