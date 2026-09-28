import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.preprocessing import (
    age_to_category,
    create_patient_record,
    create_tabular_pipe,
    get_age,
    get_transforms,
    load_dicom,
    split_train_val,
)

MONO2_PATH = "data/unit_test/images/1.2.826.0.1.3680043.8.498.10317958066544188933216299690107088692.dcm"
MONO1_PATH = "data/unit_test/images/1.2.826.0.1.3680043.8.498.42390888814664320877923477155758065418.dcm"


def test_load_dicom_returns_pil_image():
    img = load_dicom(MONO2_PATH)
    assert isinstance(img, Image.Image)


def test_load_dicom_pixel_values_in_range():
    arr = np.array(load_dicom(MONO2_PATH))
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_load_dicom_pixel_dtype_uint8():
    arr = np.array(load_dicom(MONO2_PATH))
    assert arr.dtype == np.uint8


def test_get_age_normal():
    assert get_age("19580101", "20210101") == 63


def test_get_age_same_year():
    assert get_age("20000101", "20000601") == 0


def test_get_age_century_boundary():
    assert get_age("19991231", "20010101") == 2


def test_age_to_category_falls_in_first_bin():
    bins = np.array([0, 50, 70, 90, np.inf])
    result = age_to_category(pd.Series([25.0]), bins)
    assert result[0, 0] == 0


def test_age_to_category_boundary_is_left_closed():
    bins = np.array([0, 50, 70, 90, np.inf])
    result = age_to_category(pd.Series([50.0]), bins)
    assert result[0, 0] == 1


def test_age_to_category_falls_in_last_bin():
    bins = np.array([0, 50, 70, 90, np.inf])
    result = age_to_category(pd.Series([95.0]), bins)
    assert result[0, 0] == 3


def test_tabular_pipe_no_bins_standardizes_age_to_zero_mean():
    pipe = create_tabular_pipe(bins=None)
    df = pd.DataFrame({
        "patient_age": [60.0, 70.0, 80.0, 50.0, 55.0],
        "gender": ["FEMALE", "MALE", "FEMALE", "MALE", "FEMALE"],
    })
    pipe.fit(df)
    out = pipe.transform(df)
    assert abs(out["patient_age"].mean()) < 1e-6


def test_tabular_pipe_no_bins_produces_gender_male_column():
    pipe = create_tabular_pipe(bins=None)
    df = pd.DataFrame({
        "patient_age": [60.0, 70.0],
        "gender": ["FEMALE", "MALE"],
    })
    pipe.fit(df)
    out = pipe.transform(df)
    assert "gender_MALE" in out.columns


def test_tabular_pipe_no_bins_female_encodes_to_zero():
    pipe = create_tabular_pipe(bins=None)
    df = pd.DataFrame({
        "patient_age": [60.0, 70.0],
        "gender": ["FEMALE", "MALE"],
    })
    pipe.fit(df)
    out = pipe.transform(df)
    assert out.loc[0, "gender_MALE"] == 0.0
    assert out.loc[1, "gender_MALE"] == 1.0


def test_tabular_pipe_with_bins_produces_integer_dtype():
    pipe = create_tabular_pipe(bins=[0, 50, 70, 90])
    df = pd.DataFrame({
        "patient_age": [40.0, 60.0, 80.0],
        "gender": ["FEMALE", "MALE", "FEMALE"],
    })
    pipe.fit(df)
    out = pipe.transform(df)
    assert np.issubdtype(out["patient_age"].dtype, np.integer)


def test_tabular_pipe_with_bins_category_labels_in_valid_range():
    pipe = create_tabular_pipe(bins=[0, 50, 70, 90])
    df = pd.DataFrame({
        "patient_age": [40.0, 60.0, 80.0],
        "gender": ["FEMALE", "MALE", "FEMALE"],
    })
    pipe.fit(df)
    out = pipe.transform(df)
    assert out["patient_age"].isin([0, 1, 2, 3]).all()


def test_split_train_val_sizes_sum_to_total(stratified_df):
    train, val = split_train_val(stratified_df, test_size=0.2, random_state=42)
    assert len(train) + len(val) == len(stratified_df)


def test_split_train_val_val_proportion(stratified_df):
    train, val = split_train_val(stratified_df, test_size=0.2, random_state=42)
    assert abs(len(val) / len(stratified_df) - 0.2) < 0.05


def test_split_train_val_no_uid_overlap(stratified_df):
    train, val = split_train_val(stratified_df, random_state=42)
    assert len(set(train["uid"]) & set(val["uid"])) == 0


def test_split_train_val_is_deterministic(stratified_df):
    train1, _ = split_train_val(stratified_df, random_state=42)
    train2, _ = split_train_val(stratified_df, random_state=42)
    assert list(train1["uid"]) == list(train2["uid"])


def test_get_transforms_train_includes_random_flip():
    t = get_transforms(512, type="train")
    names = [type(tr).__name__ for tr in t.transforms]
    assert "RandomHorizontalFlip" in names


def test_get_transforms_val_output_tensor_shape():
    t = get_transforms(224, type="val")
    dummy = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    out = t(dummy)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 224, 224)


def test_create_patient_record_female_sex_mapped_to_FEMALE():
    sex, _, _ = create_patient_record(MONO2_PATH)
    assert sex == "FEMALE"


def test_create_patient_record_age_is_int():
    _, age, _ = create_patient_record(MONO2_PATH)
    assert isinstance(age, int)


def test_create_patient_record_photometric_is_uppercase():
    _, _, photometric = create_patient_record(MONO2_PATH)
    assert photometric == photometric.upper()


def test_create_patient_record_mono1_photometric():
    _, _, photometric = create_patient_record(MONO1_PATH)
    assert photometric == "MONOCHROME1"
