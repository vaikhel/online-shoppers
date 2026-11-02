import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(path: str = "../data/online_shoppers_intention.csv") -> pd.DataFrame:
    """Load the Online Shoppers Intention dataset."""
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()

def get_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """Create preprocessing pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    return preprocessor
