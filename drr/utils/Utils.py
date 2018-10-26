import torch
import numpy as np

from torch.autograd import Variable
from drr.utils.BuildDict import BuildDict
from drr.utils.DataSetUtil import DataSet


class Utils:
    def __init__(self, opts):
        self.batch_size = opts['batch_size']
        self.train_path = opts['train_path']
        self.path = opts['path']

    def getGrn16SentencesAndDict(self):
        dict = (BuildDict({
            'path': self.train_path
        })).run()

        DataSetModel = DataSet({
            'path': self.path,
            'dict_dict': dict,
            'batch_size': self.batch_size
        })

        arg1List, arg2List, labelList = DataSetModel.getGrn16Sentences()

        loader = DataSetModel.getGrn16TensorDataset({
            'arg1List': Variable(torch.from_numpy(np.array(arg1List))),
            'arg2List': Variable(torch.from_numpy(np.array(arg2List))),
            'labelList': Variable(torch.from_numpy(np.array(labelList))),
        })

        return (loader, dict)
