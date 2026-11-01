import pandas as pd

def load_data(path: str = "../data/online_shoppers_intention.csv") -> pd.DataFrame:
    """Load the Online Shoppers Intention dataset."""
    df = pd.read_csv(path)
    return df
