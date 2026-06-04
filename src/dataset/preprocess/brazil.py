"""Thin wrapper — delegates to the shared airports preprocessor."""
from src.dataset.preprocess.airports import preprocess

if __name__ == '__main__':
    preprocess('brazil')
    print('Done!')
