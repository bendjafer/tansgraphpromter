import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class EuropeMiniLMDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.graph = torch.load(self.processed_file_names[0], weights_only=False)
        self.text  = pd.read_csv(self.processed_file_names[1])
        self.prompt = (
            "\nQuestion: What is the activity level of this airport? "
            "Choose one of: Low, Medium, High, or Very High.\n\nAnswer:"
        )
        self.graph_type   = 'Airport Network'
        self.num_features = 384  # all-MiniLM-L12-v2 embedding dim
        self.num_classes  = 4    # Low, Medium, High, Very High
        # GNN config for 384-dim input
        self.gnn_hidden_dim = 256
        self.gnn_out_dim    = 256
        self.gnn_num_layers = 2

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            row = self.text.iloc[index]
            return {
                'id':       int(row['node_id']),
                'label':    row['label'],
                'desc':     row['description'],
                'question': self.prompt,
            }

    @property
    def processed_file_names(self):
        return [
            'dataset/tape_europe_minilm/processed/data.pt',
            'dataset/tape_europe_minilm/processed/text.csv',
        ]

    def get_idx_split(self):
        def _load(path):
            with open(path) as f:
                return [int(line.strip()) for line in f]
        return {
            'train': _load('dataset/tape_europe_minilm/split/train_indices.txt'),
            'val':   _load('dataset/tape_europe_minilm/split/val_indices.txt'),
            'test':  _load('dataset/tape_europe_minilm/split/test_indices.txt'),
        }


if __name__ == '__main__':
    dataset = EuropeMiniLMDataset()
    print(f'graph.x shape: {dataset.graph.x.shape}')
    print(json.dumps(dataset[0], indent=4))
    split = dataset.get_idx_split()
    for k, v in split.items():
        print(f'# {k}: {len(v)}')
