from src.data_loader import load_data
from src.model import WindowedDataset, create_dataset
from src.config import WINDOW_SIZE, PRED_DISTANCE


def test_create_dataset_shapes():
    df = load_data('data1.csv', 1).iloc[:1000]
    dataset = create_dataset(df, WINDOW_SIZE, PRED_DISTANCE)
    assert isinstance(dataset, WindowedDataset)
    assert dataset.X.ndim == 3 and dataset.y.ndim == 2
    assert dataset.X.shape[1] == WINDOW_SIZE
    assert dataset.y.shape[1] == PRED_DISTANCE
    # Baseline should match horizon dimension
    assert dataset.baselines.shape == dataset.y.shape
