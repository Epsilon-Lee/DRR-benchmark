import torch

import argparse
import drr

parser = argparse.ArgumentParser("preprocess.py")

parser.add_argument('--save_data', default='data/')
parser.add_argument('--train_path', default='raw_data/train.raw.txt.tok')
parser.add_argument('--dev_path', default='raw_data/dev.raw.txt.tok')
parser.add_argument('--test_path', default='raw_data/test.raw.txt.tok')
parser.add_argument('--vocab_size', type=int, default=10000)
parser.add_argument('--data_stat_file', default='data/stat.log')

args = parser.parse_args()

if __name__ == '__main__':

    # build dict
    with open(args.train_path, 'r') as f:
        lines = f.readlines()
        words = []
        labels = []
        for line in lines:
            line_split = line.split('|||')
            arg1 = line_split[1].split()
            arg2 = line_split[2].split()
            words.extend(arg1)
            words.extend(arg2)
            labels.append(line_split[0])
        word_freq = {}
        label_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        for label in labels:
            if label in label_freq:
                label_freq[label] += 1
            else:
                label_freq[label] = 1

    print('\nRaw vocabulary size: %d' % len(word_freq))

    word_freq_lst = list(word_freq.items())
    word_freq_lst.sort(key=lambda e: e[1], reverse=True)

    label_freq_lst = list(label_freq.items())
    label_freq_lst.sort(key=lambda e: e[1], reverse=True)

    print('\nTraining label frequency')
    for label, freq in label_freq_lst:
        print('%s : %d' %(label, freq))

    word2id = {}
    id2word = {}
    for id in range(args.vocab_size):
        word2id[word_freq_lst[id][0]] = id
        id2word[id] = word_freq_lst[id][0]

    label2id = {}
    id2label = {}
    for id in range(len(label_freq_lst)):
        label2id[label_freq_lst[id][0]] = id
        id2label[id] = label_freq_lst[id][0]

    dict_dict = {
        'word2id' : word2id,
        'id2word' : id2word,
        'label2id': label2id,
        'id2label': id2label
    }
    # diction = drr.utils.Dict(
    #     word2id,
    #     id2word,
    #     label2id,
    #     id2label
    # )

    # torch.save(diction, args.save_data + 'dict.pt')

    train_ds = drr.utils.Dataset(
        args.train_path,
        dict_dict
    )

    dev_ds = drr.utils.Dataset(
        args.dev_path,
        dict_dict
    )

    test_ds = drr.utils.Dataset(
        args.test_path,
        dict_dict
    )

    torch.save(train_ds, os.path.join(args.save_data, 'train_data.pt'))
    torch.save(dev_ds, os.path.join(args.save_data, 'dev_data.pt'))
    torch.save(test_ds, os.path.join(args.save_data, 'test_data.pt'))