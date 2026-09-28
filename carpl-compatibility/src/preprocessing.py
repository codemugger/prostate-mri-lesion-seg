import pydicom, os
import pandas as pd
import numpy as np
import logging

from PIL import Image
from torchvision import transforms

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def split_train_val(tabular_df, test_size=0.1765, random_state=42):
    """Splits the tabular dataframe into training and validation sets with stratification on a combination of age, gender, 
    target (Normal, Osteopenia and Osteoporosis), patient type (A & E, Out-Patient, In-Patient) and whether they're from a polyclinic, 
    which are indicated in the 'strat_var' column.

    Args:
        tabular_df (pd.DataFrame): The input tabular dataframe containing patient records and target labels.
        test_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Controls the randomness of the split for reproducibility.

    Returns:
        tuple: A tuple containing the training and validation dataframes (train_df, val_df).
    """
    train_df, val_df = train_test_split(tabular_df, test_size=test_size, random_state=random_state, stratify=tabular_df['strat_var'])
    return train_df, val_df

def name_out(transformer, feature_names_in):
        return ["patient_age"]

def age_to_category(x, bins):
    x = x.to_numpy().ravel()
    return (
        pd.cut(x, bins=bins, labels=range(len(bins)-1), right=False)
        .astype(int)
        .reshape(-1, 1)
    )

def create_tabular_pipe(bins=None):
    """Create a tabular data preprocessing pipeline based on the provided configuration.

    Args:
        conf (Namespace): Configuration object containing preprocessing parameters.

    Returns:
        sklearn.pipeline.Pipeline: A scikit-learn pipeline for tabular data preprocessing.
    """
    if bins:
        logger.info("Using age binning with bins: {}".format(bins))
        bins = np.append(np.array(bins), [np.inf])
        age_pipe = make_column_transformer(
            (
                FunctionTransformer(age_to_category, kw_args={'bins': bins}, feature_names_out=name_out), ['patient_age']),
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')
    else:
        logger.info("Using standardised age without binning")
        age_pipe = make_column_transformer(
            (make_pipeline(
                SimpleImputer(strategy='median'), 
                StandardScaler()), 
            ['patient_age']),
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform='pandas')
    gender_pipe = make_column_transformer(
        (make_pipeline(
            SimpleImputer(strategy='most_frequent'), 
            OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)), 
        ['gender']),
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return make_pipeline(age_pipe, gender_pipe)

def get_tabular_pipe(tabular_pipe_path):
    with open(tabular_pipe_path, "rb") as f:
        tabular_pipe = pd.read_pickle(f)
    return tabular_pipe

def get_calibrator(calibrator_path):
    with open(calibrator_path, "rb") as f:
        return pd.read_pickle(f)

def load_dicom(dcm_path):
    dicom_image = pydicom.dcmread(dcm_path)
    pixel_array = dicom_image.pixel_array.astype(np.float32)
    pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
    pixel_array = np.uint8(np.round(pixel_array))
    image = Image.fromarray(pixel_array)
    return image

def get_transforms(img_size, type="train"):
    if type == "train":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Small rotation for medical images
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                fill=0
            ),
            # Medical image specific augmentations
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle brightness/contrast changes
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    
    # for validation, evaluation and inference
    else:
        return transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_age(birth_date, acqdate):
    birth_year = int(birth_date[:4])
    acquisition_year = int(acqdate[:4])
    age = acquisition_year - birth_year
    return age

def create_patient_record(file_path):
    dicom_image = pydicom.dcmread(file_path)

    sex = getattr(dicom_image, "PatientSex", None)
    age = getattr(dicom_image, "PatientAge", None)
    if age == "":
        age = None
    photometric = getattr(dicom_image, "PhotometricInterpretation", None)
    birthdate = getattr(dicom_image, "PatientBirthDate", None)
    if birthdate == "":
        birthdate = None
    acqdate = getattr(dicom_image, "AcquisitionDate", None)
    if acqdate == "":
        acqdate = None

    # checks for missing patient information and log the file paths with missing info
    # if sex is None:
    #     logger.info(f"{file_path} missing PatientSex")
    # if age is None:
    #     logger.info(f"{file_path} missing PatientAge")
    # if photometric is None:
    #     logger.info(f"{file_path} missing PhotometricInterpretation")
    # if birthdate is None:
    #     logger.info(f"{file_path} missing PatientBirthDate")
    # if acqdate is None:
    #     logger.info(f"{file_path} missing AcquisitionDate")

    if sex is not None and photometric is not None:
        if age is None and birthdate is None and acqdate is None:
            logger.info(f"{file_path} has missing age information and no birthdate or acquisition date to calculate age")
        else:
            if birthdate is not None and acqdate is not None:
                age = int(get_age(birthdate, acqdate))
            else:
                age = int(age[:3])  # Extract the age as an integer (assuming it's in the format "XXXY" where XXX is the age and Y is the unit)
        
        if sex == "M":
            sex = "MALE"
        elif sex == "F":
            sex = "FEMALE"
    else:
        logger.info(f"{file_path} has incomplete patient information")
    return sex, age, photometric

def create_tabular_data(image_dir, target_csv_path=None):
    """Creates a tabular dataframe containing patient records extracted from DICOM files in the specified folder path.
    Args:
        image_dir: Path to the directory containing DICOM files.
        target_csv_path: Path to the CSV file containing target values (optional). None if target values are not available (e.g., during inference).

    Returns:
        pd.DataFrame: A pandas DataFrame containing patient records with columns ['uid', 'patient_age', 'gender', 'photometric', 'target'].
    """
    logger.info(f"Creating tabular data from {image_dir}")

    if target_csv_path:
        target_df = pd.read_csv(target_csv_path)
        uid_list = target_df["uid"].astype(str).tolist()
    else:
        uid_list = [f[:-4] for f in os.listdir(image_dir)]

    patient_records = []

    for uid in uid_list:
        dcm_path = os.path.join(image_dir, f"{uid}.dcm")

        if not os.path.exists(dcm_path):
            logger.info(f"Missing image for {uid}")
            continue

        try:
            sex, age, photometric = create_patient_record(dcm_path)
            patient_records.append([
                uid,
                int(age),
                sex,
                photometric.upper()
            ])

        except Exception as e:
            logger.info(f"Error occurred while processing {uid}: {e}")
            continue
    
    tabular_df = pd.DataFrame(patient_records, columns=['uid', 'patient_age', 'gender', 'photometric'])

    if target_csv_path:
        tabular_df = tabular_df.merge(target_df, on='uid', how='inner')
    else:
        tabular_df['target'] = 0

    logger.info(f"Created tabular dataframe with {len(tabular_df)} records and {len(patient_records) - len(tabular_df)} errors.")

    return tabular_df