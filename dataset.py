from torch.utils.data import Dataset
import pickle
import torch

class OneStepBaselineDataset(Dataset):
    def __init__(self, filename):
        data = pickle.load(open(filename, 'rb'))
        self.features = data['features']
        self.sentence_indicators = data['sentence_indicators']
        self.ids = data['ids']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        feature = {key: torch.tensor(val[idx]).cuda() for key, val in self.features.items()}
        sentence_indicator = torch.tensor(self.sentence_indicators[idx]).cuda()
        qid = self.ids[idx]
        return feature, sentence_indicator, qid