# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from newProgress import KeepProgress
from mod_rk2 import RK2

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=50)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--optimizer', type=str, default='Adam')

args = argparser.parse_args()

file, file_len = read_file(args.filename)


def random_training_set(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

decoder = RNN(n_characters, args.hidden_size, n_characters, args.n_layers)
if args.optimizer == 'RK2':
    decoder_optimizer = RK2(decoder.parameters(), lr=args.lr)
else:
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)


criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

base_ = str(random.randint(1,100000)) + '_' + str(random.randint(1,100000))


progress = KeepProgress(decoder, args, base_)
def train(inp, target):
    # hidden = decoder.init_hidden()
    # decoder.zero_grad()
    # loss = 0

    # for c in range(args.chunk_len):
    #     output, hidden = decoder(inp[c], hidden)
    #     loss += criterion(output, target[c])

    # loss.backward()

    def closure():
        hidden = decoder.init_hidden()
        decoder.zero_grad()
        loss = 0
        for c in range(args.chunk_len):
            hidden = decoder.init_hidden()
            output, hidden = decoder(inp[c], hidden)
            loss += criterion(output, target[c])
        loss.backward()
        return loss

    loss = decoder_optimizer.step(closure)
    progress.train_progress({'train_loss':loss, 'train_accuracy':0})
    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '_' + args.optimizer + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        loss = train(*random_training_set(args.chunk_len))
        loss_avg += loss

        progress.test_progress({'test_loss':loss_avg, 'test_accuracy':0})

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print(generate(decoder, 'Wh', 100), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

