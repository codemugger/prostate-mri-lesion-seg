import numpy as np
import pandas as pd
import pytest

MONO2_PATH = "data/unit_test/images/1.2.826.0.1.3680043.8.498.10317958066544188933216299690107088692.dcm"
MONO1_PATH = "data/unit_test/images/1.2.826.0.1.3680043.8.498.42390888814664320877923477155758065418.dcm"


@pytest.fixture
def stratified_df():
    """DataFrame with the strat_var column required by split_train_val."""
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        "uid": [str(i) for i in range(n)],
        "patient_age": np.random.uniform(50, 90, n),
        "gender": np.random.choice(["MALE", "FEMALE"], n),
        "target": np.random.choice([0, 1], n),
        "strat_var": np.random.choice(["A", "B", "C", "D"], n),
    })


@pytest.fixture
def post_pipeline_df():
    """Post-tabular-pipeline DataFrame as expected by OsteoDataset.

    UIDs correspond to real files already in data/unit_test/images/.
    """
    return pd.DataFrame({
        "uid": [
            "1.2.826.0.1.3680043.8.498.10317958066544188933216299690107088692",
            "1.2.826.0.1.3680043.8.498.10637030570672990824435329439158837177",
        ],
        "patient_age": [0.5, -0.3],
        "gender_MALE": [0.0, 0.0],
        "photometric": ["MONOCHROME2", "MONOCHROME2"],
        "target": [0, 0],
    })


@pytest.fixture
def simple_transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(64),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
