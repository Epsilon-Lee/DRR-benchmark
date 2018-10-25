import numpy as np
import torch.utils.data as Data


class DataSet:
    def __init__(self, opts):
        self.path = opts['path']
        self.batch_size = opts['batch_size']
        self.vocab_size = len(opts['dict_dict']['word2id'])
        self.word2id = opts['dict_dict']['word2id']
        self.label2id = opts['dict_dict']['label2id']

    def getGrn16TensorDataset(self, params):
        torch_dataset = Data.TensorDataset(params['arg1List'], params['arg2List'], params['labelList'])
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2  # set multi-work num read data
        )

        return loader

    def getGrn16Sentences(self):
        with open(self.path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            arg1List = []
            arg2List = []
            labelList = []
            for line in lines:
                label, arg1, arg2 = self.formateArg(line)
                # The subscript of the word corresponding to the word of parameter one
                arg1WordIDList = []
                # The subscript of the word corresponding to the word of the second parameter
                arg2WordIDList = []
                # Loop each word of the arg1 sentence, find the corresponding subscript of the word in word2id, as an element of arg1WordIDList
                for arg1Item in arg1:
                    if (str(arg1Item) in self.word2id.keys()):
                        arg1WordIDList.append(self.word2id[str(arg1Item)])
                # Loop each word of the arg2 sentence, find the corresponding subscript of the word in word2id, as an element of arg2WordIDList
                for arg2Item in arg2:
                    if (str(arg2Item) in self.word2id.keys()):
                        arg2WordIDList.append(self.word2id[str(arg2Item)])

                # Fill arg1WordIDList to a length of 50
                arg1WordIDList = self.resizeList(arg1WordIDList, 50)

                arg1List.extend(arg1WordIDList)

                # Fill arg2WordIDList to a length of 50
                arg2WordIDList = self.resizeList(arg2WordIDList, 50)
                arg2List.extend(arg2WordIDList)

                labelList.append(self.label2id[label])

            return (arg1List, arg2List, labelList)

    def formateArg(self, line):
        line_split = line.split('|||')
        label = line_split[0]
        arg1 = line_split[1].split()
        arg2 = line_split[2].split()

        return (label, arg1, arg2)

    def resizeList(self, list, dim):
        list = np.array(list).reshape((1, -1))
        list = list.copy()
        list.resize((1, dim))

        return list
