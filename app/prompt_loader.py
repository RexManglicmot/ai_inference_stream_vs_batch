# app/prompt_loader.py

from pathlib import Path
import pandas as pd

# It is DataFrame not Dataframe...missing the capital F

def load_prompts(path: str = "data/prompts.csv") -> pd.DataFrame:
    """
    Load prompts from CSV and do a quick on the columns
    Expected coumns: prompt_id and prompt_text
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Promots file not found: {p.resolve()}")
    
    df = pd.read_csv(p)
    required = {"prompt_id", "prompt_text"}
    if not required.issubset(df.columns):                   #subset not subet
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columsn in promots.csv: {missing}")
    return df

if __name__ == "__main__":
    # Smoke test: prints shape and first row
    df = load_prompts()
    print(df.shape)
    if len(df):
        print(df.iloc[0].to_dict())          # to_dict() not todict()

# WORKS!