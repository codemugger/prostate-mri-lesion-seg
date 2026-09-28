import pandas as pd
import pytest
import torch

from src.dataloader import OsteoDataset

IMAGE_DIR = "data/unit_test/images/"

MONO1_UID = "1.2.826.0.1.3680043.8.498.42390888814664320877923477155758065418"

def test_dataset_len_equals_dataframe_row_count(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms)
    assert len(ds) == len(post_pipeline_df)


def test_dataset_img_paths_end_with_dcm(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms)
    assert all(p.endswith(".dcm") for p in ds.img_paths)


def test_dataset_img_paths_contain_uid(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms)
    for uid, path in zip(post_pipeline_df["uid"], ds.img_paths):
        assert uid in path


def test_dataset_train_mode_returns_four_tuple(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms, type="train")
    assert len(ds[0]) == 4  # image, tabular, filename, label


def test_dataset_inference_mode_returns_three_tuple(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms, type="inference")
    assert len(ds[0]) == 3  # image, tabular, filename


def test_dataset_image_output_is_tensor(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms, type="train")
    image, _, _, _ = ds[0]
    assert isinstance(image, torch.Tensor)


def test_dataset_tabular_features_shape_is_two(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms, type="train")
    _, tabular, _, _ = ds[0]
    assert tabular.shape == (2,)


def test_dataset_tabular_features_dtype_is_float32(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms, type="train")
    _, tabular, _, _ = ds[0]
    assert tabular.dtype == torch.float32


def test_dataset_filename_is_string(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms, type="train")
    _, _, filename, _ = ds[0]
    assert isinstance(filename, str)


def test_dataset_label_matches_dataframe(post_pipeline_df, simple_transforms):
    ds = OsteoDataset(IMAGE_DIR, post_pipeline_df, simple_transforms, type="train")
    _, _, _, label = ds[0]
    assert label == post_pipeline_df["target"].iloc[0]
    

def test_dataset_monochrome1_image_differs_from_monochrome2(simple_transforms):
    """The same physical DICOM treated as MONOCHROME1 should produce a pixel-inverted
    tensor compared to when it is treated as MONOCHROME2."""
    df_mono1 = pd.DataFrame({
        "uid": [MONO1_UID],
        "patient_age": [0.5],
        "gender_MALE": [0.0],
        "photometric": ["MONOCHROME1"],
        "target": [1],
    })
    df_mono2 = df_mono1.copy()
    df_mono2["photometric"] = "MONOCHROME2"

    ds_mono1 = OsteoDataset(IMAGE_DIR, df_mono1, simple_transforms, type="train")
    ds_mono2 = OsteoDataset(IMAGE_DIR, df_mono2, simple_transforms, type="train")

    img_mono1, _, _, _ = ds_mono1[0]
    img_mono2, _, _, _ = ds_mono2[0]

    assert not torch.equal(img_mono1, img_mono2)
