from __future__ import print_function
import argparse
import os
import random
import sys

sys.path.append(os.getcwd())
import pickle
import time
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from misc.utils import adjust_learning_rate
from misc.dataLoader import Data
import misc.model as model
import datetime

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--input_video_h5', default='../../dataset_video/data/video_feat.hdf5',
                    help='path to dataset, now hdf5 file')
parser.add_argument('--input_text_h5', default='../../dataset_text/TurnsL5/text_features.hdf5',
                    help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='../../dataset_text/TurnsL5/dictionary.json',
                    help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--save_iter', type=int, default=2, help='number of epochs to save for')
parser.add_argument('--encoder', default='G_QIH_VGG', help='what encoder to use.')
parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', default=1000, help='number of image split out as validation set.')
parser.add_argument('--model_path_D', default='', help='folder to output images and model checkpoints')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--start_epoch', type=int, default=0, help='start of epochs to train for')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate for, default=0.00005')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose', action='store_true', help='show the sampled caption')
parser.add_argument('--conv_feat_size', type=int, default=4096, help='input batch size')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--mos', action='store_true', help='whether to use Mixture of Softmaxes layer')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--log_interval', type=int, default=50, help='how many iterations show the log info')
parser.add_argument('--monte_carlo_simulations', default=100, help='Number of sampling')
parser.add_argument('--ans_classes', default=100, help='Number of ans class for discriminitor')
parser.add_argument('--turn_length', default=5, help='Number of turns in the dialog')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    model_path = opt.model_path
    opt = checkpoint['opt']
    opt.start_epoch = checkpoint['epoch']
    opt.model_path = model_path
    opt.batchSize = 128
    opt.niter = 100
else:
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' % (t.day, t.month, t.hour)
    save_path = os.path.join(opt.outf, opt.encoder + '.' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

writer = SummaryWriter(save_path)
####################################################################################
# Data Loader
####################################################################################


dataset_train = Data(input_video_file=opt.input_video_h5, input_text_file=opt.input_text_h5, input_json=opt.input_json,
                     data_split='Train')
dataset_val = Data(input_video_file=opt.input_video_h5, input_text_file=opt.input_text_h5, input_json=opt.input_json,
                   data_split='Val')
dataset_test = Data(input_video_file=opt.input_video_h5, input_text_file=opt.input_text_h5, input_json=opt.input_json,
                    data_split='Test')

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True,
                                               num_workers=int(opt.workers))
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=int(opt.workers))
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Building the Model
####################################################################################

out_features_size = 1
vocab_size = dataset_train.vocab_size
print('vocab_size = ', vocab_size)
print('init Generative and Discriminator model...')
# define Model
NO_OF_DIALOG_TURNS = 5  # dataset_train.dialog_length
netE_text = model._netE_text(vocab_size, opt.ninp, opt.nhid, out_features_size, opt.dropout)
critD = nn.BCELoss()  # thisis discriminitor losss

if opt.cuda:  # ship to cuda, if has GPU
    netE_text.cuda()
    critD.cuda()

if opt.model_path_G != '':
    print('Loading Generative model...')
    netE_text.load_state_dict(checkpoint['netE_text'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Total number paramet in netE_text = ', count_parameters(netE_text))


#########################################################################
################    Train thr Model     #################################
#########################################################################

# training function
def train(epoch):
    netE_text.train()

    lr = adjust_learning_rate(optimizerLM, epoch, opt.lr)

    average_loss = 0
    average_loss_temp = 0
    count = 0

    # while i < len(dataloader_train):
    for i, data in enumerate(dataloader_train):
        var_params = {
            'volatile': False,
            'requires_grad': False,
        }
        dialog_turns = Variable(data['dialog_turns'].type(torch.FloatTensor).cuda(async=True), **var_params)
        dialog_turns_lengths = Variable(data['turn_lengths'].type(torch.FloatTensor).cuda(async=True), **var_params)
        label = Variable(data['label'].type(torch.FloatTensor).cuda(async=True), **var_params)
        dialog_id = data['dialog_id']

        # print("Printing shapes..")
        # print('dialog_turns',dialog_turns.shape) #dialog_turns torch.Size([32, 5, 42])
        # print('label',label.shape) #label torch.Size([32])
        # print('dialog_id',dialog_id) # it is list of batch size
        batch_size = dialog_turns.size(0)
        num_turns = dialog_turns.size(1)
        sequence_length = dialog_turns.size(2)

        text_input = dialog_turns.view(-1, sequence_length)  # check with same variable
        dialog_turns_lengths = dialog_turns_lengths.view(-1)
        logit = netE_text(text_input, dialog_turns_lengths, batch_size)
        g_loss = critD(logit, label.view(-1, 1))

        # do backward.
        netE_text.zero_grad()

        g_loss.backward()
        optimizerLM.step()
        average_loss += g_loss.item()
        average_loss_temp += g_loss.item()
        if i % opt.log_interval == 0:
            print("step {} / {} (epoch {}), g_loss {:.5f},lr = {:.6f}".format(i, len(dataloader_train), epoch,
                                                                              average_loss_temp / opt.log_interval, lr))
            average_loss_temp = 0
        count = count + 1
    average_loss /= count

    return average_loss, lr


def val():
    # global text_input
    netE_text.eval()
    # netC.eval()
    y_true = []
    y_pred = []
    y_prob = []

    data_iter_val = iter(dataloader_val)
    i = 0
    average_loss = 0
    count = 0
    final_f1_score = 0
    final_recall = 0
    final_precision = 0
    final_accuracy_score = 0
    for i, data in enumerate(dataloader_val):
        var_params = {
            'volatile': True,
            'requires_grad': False,
        }
        dialog_turns = Variable(data['dialog_turns'].type(torch.FloatTensor).cuda(async=True), **var_params)
        dialog_turns_lengths = Variable(data['turn_lengths'].type(torch.FloatTensor).cuda(async=True), **var_params)
        label = Variable(data['label'].type(torch.FloatTensor).cuda(async=True), **var_params)
        dialog_id = data['dialog_id']

        batch_size = dialog_turns.size(0)
        num_turns = dialog_turns.size(1)
        sequence_length = dialog_turns.size(2)
        text_input = dialog_turns.view(-1, sequence_length)
        dialog_turns_lengths = dialog_turns_lengths.view(-1)
        logit = netE_text(text_input, dialog_turns_lengths, batch_size)
        g_loss = critD(logit, label.view(-1, 1))

        average_loss += g_loss.item()
        count += 1
        y_prob.append(logit)
        logit[logit > 0.5] = 1
        logit[logit <= 0.5] = 0
        y_true.append(label.view(-1, 1))
        y_pred.append(logit)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_prob = torch.cat(y_prob)
    y_true = y_true.cpu().data.numpy()
    y_pred = y_pred.cpu().data.numpy()
    y_prob = y_prob.cpu().data.numpy()
    final_accuracy_score = accuracy_score(y_true, y_pred)
    final_f1_score = f1_score(y_true, y_pred)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_roc_auc_score = roc_auc_score(y_true, y_prob)
    final_roc_curve = roc_curve(y_true, y_prob, pos_label=1)
    average_loss /= count

    # return average_loss
    return average_loss, final_accuracy_score, final_f1_score, final_recall, final_precision, final_roc_curve, final_roc_auc_score


def Test():
    # global text_input
    netE_text.eval()
    # netC.eval()
    y_true = []
    y_pred = []
    y_prob = []

    data_iter_test = iter(dataloader_test)
    i = 0
    average_loss = 0
    count = 0
    final_f1_score = 0
    final_recall = 0
    final_precision = 0
    final_accuracy_score = 0
    for i, data in enumerate(dataloader_test):
        var_params = {
            'volatile': True,
            'requires_grad': False,
        }
        dialog_turns = Variable(data['dialog_turns'].type(torch.FloatTensor).cuda(async=True), **var_params)
        dialog_turns_lengths = Variable(data['turn_lengths'].type(torch.FloatTensor).cuda(async=True), **var_params)
        label = Variable(data['label'].type(torch.FloatTensor).cuda(async=True), **var_params)
        dialog_id = data['dialog_id']
        batch_size = dialog_turns.size(0)
        num_turns = dialog_turns.size(1)
        sequence_length = dialog_turns.size(2)
        text_input = dialog_turns.view(-1, sequence_length)  # check with same variable
        dialog_turns_lengths = dialog_turns_lengths.view(-1)
        logit = netE_text(text_input, dialog_turns_lengths, batch_size)
        g_loss = critD(logit, label.view(-1, 1))

        average_loss += g_loss.item()
        count += 1
        y_prob.append(logit)
        logit[logit > 0.5] = 1
        logit[logit <= 0.5] = 0
        y_true.append(label.view(-1, 1))
        y_pred.append(logit)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_prob = torch.cat(y_prob)
    y_true = y_true.cpu().data.numpy()
    y_pred = y_pred.cpu().data.numpy()
    y_prob = y_prob.cpu().data.numpy()
    final_accuracy_score = accuracy_score(y_true, y_pred)
    final_f1_score = f1_score(y_true, y_pred)
    final_precision = precision_score(y_true, y_pred)
    final_recall = recall_score(y_true, y_pred)
    final_roc_auc_score = roc_auc_score(y_true, y_prob)
    final_roc_curve = roc_curve(y_true, y_prob, pos_label=1)
    average_loss /= count

    # return average_loss
    return average_loss, final_accuracy_score, final_f1_score, final_recall, final_precision, final_roc_curve, final_roc_auc_score


####################################################################################
# Main
####################################################################################

optimizerLM = optim.Adam([{'params': netE_text.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))
history = []
history_pickle = []
train_his = {}

for epoch in range(opt.start_epoch + 1, opt.niter):
    t = time.time()
    train_loss_lm, lr = train(epoch)

    print('Evaluating ... ')
    # val_loss = val()
    val_loss, final_accuracy_score, final_f1_score, final_recall, final_precision, final_roc_curve, final_roc_auc_score = val()

    print('Epoch: %d learningRate: %4f train loss :  %4f val loss: %4f Time: %3f' % (
        epoch, lr, train_loss_lm, val_loss, time.time() - t))
    print(
        'Val Result Epoch: %d final_accuracy_score :  %4f final_f1_score: %4f final_recall: %4f final_precision: %4f final_roc_auc_score: %4f' % (
            epoch, final_accuracy_score, final_f1_score, final_recall, final_precision, final_roc_auc_score))

    writer.add_scalars("Loss", {"Train_loss": train_loss_lm, "Val_loss": val_loss}, epoch)
    writer.add_scalar("Accuracy", final_accuracy_score, epoch)
    writer.add_scalar("f1_score", final_f1_score, epoch)
    writer.add_scalar("precision", final_precision, epoch)
    writer.add_scalar("recall", final_recall, epoch)
    writer.add_scalar("roc_auc_score", final_roc_auc_score, epoch)

    print('Test Evaluating ... ')
    test_loss, test_final_accuracy_score, test_final_f1_score, test_final_recall, test_final_precision, test_final_roc_curve, test_final_roc_auc_score = Test()
    print(
        'Test Result Epoch: %d final_Test_Loss :  %4f final_accuracy_score :  %4f final_f1_score: %4f final_recall: %4f final_precision: %4f final_roc_auc_score: %4f' % (
            epoch, test_loss, test_final_accuracy_score, test_final_f1_score, test_final_recall, test_final_precision,
            test_final_roc_auc_score))

    writer.add_scalars("final Loss", {"Train_loss": train_loss_lm, "Val_loss": val_loss, "Test_loss": test_loss}, epoch)
    writer.add_scalars("final Accuracy",
                       {"val Accuracy": final_accuracy_score, "test Accuracy": test_final_accuracy_score}, epoch)

    train_his = {'train_loss': train_loss_lm}
    val_his = {'val_loss': val_loss, 'accuracy_score': final_accuracy_score, "f1_score": final_f1_score,
               "recall": final_recall, "precision": final_precision, "roc_auc_score": final_roc_auc_score}
    val_his_pickle = {'val_loss': val_loss, 'accuracy_score': final_accuracy_score, "f1_score": final_f1_score,
                      "recall": final_recall, "precision": final_precision, "roc_auc_score": final_roc_auc_score,
                      "roc_curve": final_roc_curve}
    history.append({'epoch': epoch, 'train': train_his, 'val_his': val_his})
    history_pickle.append({'epoch': epoch, 'train': train_his, 'val_his': val_his_pickle})

    # saving the model.
    if epoch % opt.save_iter == 0:
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'netE_text': netE_text.state_dict(),
                    # 'netC': netC.state_dict(),
                    },
                   '%s/epoch_%d.pth' % (save_path, epoch))
        json.dump(history, open('%s/log.json' % (save_path),
                                'w'))  # this is the problem with pytorch version 4 it will work on version 3 or less
        pickle.dump(history_pickle, open('%s/log.pickle' % (save_path), 'wb'))

writer.close()
