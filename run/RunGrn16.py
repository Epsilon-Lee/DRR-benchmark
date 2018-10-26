import torch
import torch.nn as nn
from torch.autograd import Variable

# customize
import drr.models as DrrModels
import drr.utils as DrrUtils

# Hyper Parameters
EPOCH = 100  # How many times to train the entire batch of data
BATCH_SIZE = 32  # how many samples per batch to load
LR = 1e-3  # Learning rate


class RunGrn16:
    def __init__(self, opts):
        self.train_path = opts['train_path']
        self.test_path = opts['test_path']
        self.model_path = opts['model_path']

    def runTrain(self):
        torch.manual_seed(1)

        loader, dict = (DrrUtils.Utils({
            'train_path': self.train_path,
            'path': self.train_path,
            'batch_size': BATCH_SIZE
        })).getGrn16SentencesAndDict()

        Grn16Model = DrrModels.GRN16({
            'vocab_size': len(dict['word2id']),
            'r': 2
        })

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(Grn16Model.parameters(), lr=LR)

        for epoch in range(EPOCH):
            print('epoch: {}'.format(epoch + 1))
            print('****************************')
            num = 0
            running_loss = 0

            for step, (arg1List, arg2List, labelList) in enumerate(loader):
                arg1 = Variable(arg1List.long())
                arg2 = Variable(arg2List.long())
                label = Variable(labelList.long())

                # forward
                out = Grn16Model((arg1, arg2))
                loss = criterion(out, label)
                running_loss += loss.data.item()

                print('loss')
                print(loss.data.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num += loader.__len__()

            print('Loss: {:.6f}'.format(running_loss / num))

        # Save model
        torch.save(Grn16Model, self.model_path)

    def runTest(self):
        # prediction
        loader, dict = (DrrUtils.Utils({
            'train_path': self.train_path,
            'path': self.test_path,
            'batch_size': BATCH_SIZE
        })).getGrn16SentencesAndDict()

        id2label = dict['id2label']

        Grn16Model = torch.load(self.model_path)

        num = 0
        true_count = 0
        for step, (arg1List, arg2List, labelList) in enumerate(loader):
            arg1 = Variable(arg1List.long())
            arg2 = Variable(arg2List.long())
            labelList = labelList.numpy()

            out = Grn16Model((arg1, arg2))
            # Axis = 0 by column; axis = 1 by line
            _, predict_label = torch.max(out, 1)

            for i in predict_label.numpy():
                if (id2label[i] == id2label[labelList[i]]):
                    true_count += 1
                print(id2label[i] + '-' + id2label[labelList[i]])

            num += loader.__len__()

        print('Correct rate：{:.6f}%'.format((true_count / num) * 100))
