import numpy as np
import os
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time

from parser import BiAffineParser
from data_utils import make_dataset, batch_loader, split_train_test

"""
Training script for BiAffine Depenency Parser
"""

def train(n_epochs=10):
    data_file = '../data/train-stanford-raw.conll'
    # if vocab_file is given (ie for pretrained wordvectors), use x2i and i2x from this file.
    # If not given, create new vocab file in ../data
    vocab_file = None

    log_folder = '../logs'
    model_folder = '../models'
    model_name = 'wsj_3'

    model_file = os.path.join(model_folder, model_name+'_{}.model')
    log_file = open(os.path.join(log_folder, model_name+'.csv'), 'w', 1)
    print('epoch,train_loss,val_loss,arc_acc,lab_acc', file=log_file)

    batch_size = 64
    prints_per_epoch = 10
    n_epochs *= prints_per_epoch

    # load data
    print('loading data...')
    data, x2i, i2x = make_dataset(data_file)

    if not vocab_file:
        with open('../data/vocab_{}.pkl'.format(model_name), 'wb') as f:
            pickle.dump((x2i, i2x), f)

    # make train and val batch loaders
    train_data, val_data = split_train_test(data)
    print('# train sentences', len(train_data))
    print('# val sentences', len(val_data))
    train_loader = batch_loader(train_data, batch_size)
    val_loader = batch_loader(val_data, batch_size, shuffle=False)


    print('creating model...')
    # make model
    model = BiAffineParser(word_vocab_size=len(x2i['word']), word_emb_dim=100,
                           pos_vocab_size=len(x2i['tag']), pos_emb_dim=28, emb_dropout=0.33,
                           lstm_hidden=512, lstm_depth=3, lstm_dropout=.33,
                           arc_hidden=256, arc_depth=1, arc_dropout=.33, arc_activation='ReLU',
                           lab_hidden=128, lab_depth=1, lab_dropout=.33, lab_activation='ReLU',
                           n_labels=len(x2i['label']))
    print(model)
    model.cuda()
    base_params, arc_params, lab_params = model.get_param_groups()

    opt = Adam([
        {'params': base_params, 'lr':2e-3},
        {'params': arc_params, 'lr':2e-3},
        {'params': lab_params, 'lr':1e-4},
    ], betas=[.9, .9])
    sched = ReduceLROnPlateau(opt, threshold=1e-3, patience=8, factor=.4, verbose=True)

    n_train_batches = int(len(train_data) / batch_size)
    n_val_batches =  int(len(val_data) / batch_size)
    batches_per_epoch = int(n_train_batches / prints_per_epoch)

    for epoch in range(n_epochs):
        t0 = time.time()

        # Training
        train_loss = 0
        model.train()
        for i in range(batches_per_epoch):
            opt.zero_grad()

            # Load batch
            words, tags, arcs, lengths = next(train_loader)
            words = words.cuda()
            tags = tags.cuda()

            # Forward
            S_arc, S_lab = model(words, tags, lengths=lengths)

            # Calculate loss
            arc_loss = get_arc_loss(S_arc, arcs)
            lab_loss = get_label_loss(S_lab, arcs)
            loss = arc_loss + .025 * lab_loss
            train_loss += arc_loss.data[0] + lab_loss.data[0]

            # Backward
            loss.backward()
            opt.step()

        train_loss /= batches_per_epoch

        # Evaluation
        val_loss = 0
        arc_acc = 0
        lab_acc = 0
        model.eval()
        for i in range(n_val_batches):
            words, tags, arcs, lengths = next(val_loader)
            words = words.cuda()
            tags = tags.cuda()

            S_arc, S_lab = model(words, tags, lengths=lengths)

            arc_loss = get_arc_loss(S_arc, arcs)
            lab_loss = get_label_loss(S_lab, arcs)
            loss = arc_loss + lab_loss

            val_loss += arc_loss.data[0] + lab_loss.data[0]
            arc_acc += get_arc_accuracy(S_arc, arcs)
            lab_acc += get_label_accuracy(S_lab, arcs)

        val_loss /= n_val_batches
        arc_acc /= n_val_batches
        lab_acc /= n_val_batches
        epoch_time = time.time() - t0

        print('epoch {:.1f}\t train_loss {:.3f}\t val_loss {:.3f}\t arc_acc {:.3f}\t lab_acc {:.3f}\t time {:.1f} sec'.format(
            epoch/prints_per_epoch, train_loss, val_loss, arc_acc, lab_acc, epoch_time
        ), end="\r")

        print('{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(
            epoch/prints_per_epoch,train_loss, val_loss, arc_acc, lab_acc
        ), file=log_file)

        sched.step(val_loss)
    
    print('Done!')
    torch.save(model, model_file.format(val_loss))
    log_file.close()


def get_arc_loss(S_arc, arcs):
    """
    S_arc is a tensor of [batch, heads, deps]
    Arcs is a np array with columns [batch_idx, head, dep, label]
    Calculates softmax over columns in S_arc, S_arc[b,:,i] = P(head | dep=i)
    """
    logits = S_arc.cpu().transpose(-1, -2)[arcs[:,0], arcs[:,2], :]
    heads = Variable(torch.from_numpy(arcs[:, 1]))
    return F.cross_entropy(logits, heads)


def get_label_loss(S_label, arcs):
    """
    S_label is a tensor of shape [batch, n_labels, heads, deps]
    arc_labels is a list of tuples (batch_idx, head_idx, dep_idx, label)
    Calculates softmax over second dimension of S_label, 
    S_label[b, :, i, j] = P(label | head=i, dep=j).
    """
    logits = S_label.cpu().permute(0, 2, 3, 1)[arcs[:,0], arcs[:,1], arcs[:,2], :]
    labels = Variable(torch.from_numpy(arcs[:,3]))
    return F.cross_entropy(logits, labels)


def get_arc_accuracy(S_arc, arcs):
    heads = torch.from_numpy(arcs[:, 1])
    logits = S_arc.cpu().transpose(-1, -2)[arcs[:,0], arcs[:,2], :]
    preds = logits.data.max(1)[1].type(type(heads))
    correct = preds.eq(heads).sum()
    return correct / len(arcs)


def get_label_accuracy(S_label, arcs):
    labels = torch.from_numpy(arcs[:,3])
    logits = S_label.cpu().permute(0, 2, 3, 1)[arcs[:,0], arcs[:,1], arcs[:,2], :]
    preds = logits.data.max(1)[1].type(type(labels))
    correct = preds.eq(labels).sum()
    return correct / len(arcs)


if __name__ == "__main__":
    train()