import run

if __name__ == '__main__':
    (run.RunGrn16({
        'train_path': 'raw_data/train.raw.txt',
        'test_path': 'raw_data/test.raw.txt',
        'model_path': 'saved_models/Grn16Model.pkl'
    })).runTrain()
