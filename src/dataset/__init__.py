from src.dataset.cora import CoraDataset
from src.dataset.citeseer import CiteseerDataset
from src.dataset.pubmed import PubmedDataset
from src.dataset.arxiv import ArxivDataset
from src.dataset.products import ProductsDataset
from src.dataset.brazil import BrazilDataset
from src.dataset.europe import EuropeDataset


load_dataset = {
    'cora': CoraDataset,
    'citeseer': CiteseerDataset,
    'pubmed': PubmedDataset,
    'arxiv': ArxivDataset,
    'products': ProductsDataset,
    'brazil': BrazilDataset,
    'europe': EuropeDataset,
}
