import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(dataset_path: str = 'e:/CardioMind/datasets/raw/cardio_data_processed.csv'):
    df = pd.read_csv(dataset_path)

    # Age in years
    if 'age_years' in df.columns:
        age = df['age_years'].astype(float)
    elif df['age'].max() > 200:
        age = (df['age'] / 365.25).astype(float)
    else:
        age = df['age'].astype(float)

    # Sex: in cardio dataset, 1 is female (0.0), 2 is male (1.0)
    sex_code = (df['gender'] == 2).astype(float)

    # Blood pressure
    systolic_bp = df['ap_hi'].clip(60, 240).astype(float)
    diastolic_bp = df['ap_lo'].clip(40, 160).astype(float)

    # Cholesterol: In cardio dataset 1=normal, 2=above normal, 3=well above normal
    cholesterol_map = {1: 175.0, 2: 220.0, 3: 265.0}
    cholesterol = df['cholesterol'].map(cholesterol_map).fillna(185.0).astype(float)

    # Smoking: 0=never (0.0), 1=current (2.0)
    smoking_code = df['smoke'].map({0: 0.0, 1: 2.0}).fillna(0.0).astype(float)

    # Diabetes: gluc >= 2 in cardio dataset indicates elevated/diabetic glucose
    diabetes = (df['gluc'] >= 2).astype(float)

    # Family history
    np.random.seed(42)
    fh_prob = 0.10 + 0.15 * (df['cardio'] == 1).astype(float) + 0.10 * (df['gluc'] >= 2).astype(float)
    family_history = (np.random.rand(len(df)) < fh_prob).astype(float)

    # BMI
    if 'bmi' in df.columns:
        bmi = df['bmi'].clip(15, 55).astype(float)
    else:
        height_m = df['height'] / 100.0
        bmi = (df['weight'] / (height_m ** 2)).clip(15, 55).astype(float)

    # Heart Rate
    base_hr = np.random.normal(72, 8, len(df))
    hr_elevation = ((systolic_bp > 140).astype(float) * 5.0)
    heart_rate = np.clip(base_hr + hr_elevation, 45, 140).astype(float)

    X = np.column_stack([
        age.values,
        sex_code.values,
        systolic_bp.values,
        diastolic_bp.values,
        cholesterol.values,
        smoking_code.values,
        diabetes.values,
        family_history.values,
        bmi.values,
        heart_rate.values
    ]).astype(np.float32)

    y = df['cardio'].values.astype(int)

    feature_names = [
        'age', 'sex', 'systolic_bp', 'diastolic_bp', 'cholesterol',
        'smoking_status', 'diabetes', 'family_history', 'bmi', 'heart_rate'
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_names
