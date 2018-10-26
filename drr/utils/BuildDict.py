class BuildDict:
    def __init__(self, opts):
        self.path = opts['path']
        self.word_freq = {}
        self.label_freq = {}
        self.word2id = {}
        self.id2word = {}
        self.label2id = {}
        self.id2label = {}
        self.word_freq_list = []
        self.label_freq_list = []

    def run(self):
        # Assembly word dictionary, label dictionary
        self.formateSentences()
        # Create word vector
        self.createFreqList()
        # Word vector converted to dictionary
        self.word2id, self.id2word = self.listToDict(self.word_freq_list)
        self.label2id, self.id2label = self.listToDict(self.label_freq_list)

        dict_dict = {
            'word2id': self.word2id,
            'id2word': self.id2word,
            'label2id': self.label2id,
            'id2label': self.id2label
        }

        return dict_dict

    # Assembly word dictionary, label dictionary
    def formateSentences(self):
        with open(self.path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            words = []
            labels = []
            for line in lines:
                label, arg1, arg2 = self.formateArg(line)
                words.extend(arg1)
                words.extend(arg2)
                labels.append(label)

            self.createFreq(words, labels)

    # Format text data, split strings by '|||'
    def formateArg(self, line):
        line_split = line.split('|||')
        label = line_split[0]
        arg1 = line_split[1].split()
        arg2 = line_split[2].split()

        return (label, arg1, arg2)

    # Create a word frequency dictionary
    def createFreq(self, words, labels):
        self.word_freq = self.countItem(words)
        self.label_freq = self.countItem(labels)

    # Convert a list to a word frequency dictionary
    def countItem(self, list):
        dict = {}
        for item in list:
            if item in dict:
                dict[item] += 1
            else:
                dict[item] = 1

        return dict

    # Create word vector
    def createFreqList(self):
        self.word_freq_list = self.sortList(self.word_freq)
        self.label_freq_list = self.sortList(self.label_freq)
        for label, freq in self.label_freq_list:
            print('%s : %d' % (label, freq))

    # Sort list
    def sortList(self, dict):
        dealList = list(dict.items())
        dealList.sort(key=lambda e: e[1], reverse=True)

        return dealList

    # Convert list to dictionary
    def listToDict(self, list):
        item2id = {}
        id2item = {}
        for id in range(len(list)):
            item2id[list[id][0]] = id
            id2item[id] = list[id][0]

        return (item2id, id2item)
