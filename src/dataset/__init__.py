from src.dataset.cora import CoraDataset
from src.dataset.citeseer import CiteseerDataset
from src.dataset.pubmed import PubmedDataset
from src.dataset.arxiv import ArxivDataset
from src.dataset.products import ProductsDataset
from src.dataset.brazil import BrazilDataset
from src.dataset.europe import EuropeDataset
from src.dataset.brazil_minilm import BrazilMiniLMDataset
from src.dataset.europe_minilm import EuropeMiniLMDataset


load_dataset = {
    'cora': CoraDataset,
    'citeseer': CiteseerDataset,
    'pubmed': PubmedDataset,
    'arxiv': ArxivDataset,
    'products': ProductsDataset,
    'brazil': BrazilDataset,
    'europe': EuropeDataset,
    'brazil_minilm': BrazilMiniLMDataset,
    'europe_minilm': EuropeMiniLMDataset,
}
