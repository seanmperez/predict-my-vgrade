import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import KNNImputer

def mask_extreme_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes extreme values for climber features.
    """

    # Restrict height within 90-250 cm (3-8.5 ft)
    df['height'] = df['height'].where((df['height'] >= 90) & (df['height']<= 250))

    # Restrict weight within 20-180 kg (45-400 lbs)
    df['weight'] = df['weight'].where((df['weight'] >= 20) & (df['weight']<= 400))

    # Restrict year climbing with 0-70 years
    df['years_climbing'] = df['years_climbing'].where((df['years_climbing'] >= 0) & (df['years_climbing'] <= 70))

    # Restrict age to between 10-80 years
    df['age'] = df['age'].where((df['age'] >= 10) & (df['age'] <= 80))

    return df

def row_level_missing(df: pd.DataFrame, add_column = False,plot = False):
    """
    Find if rows with missing values have many missing.
    """

    missing_mask = df.isnull().sum(axis=1)

    if add_column:
        return missing_mask
    
    num_features_na_count = missing_mask.value_counts()

    clean_df = num_features_na_count.reset_index().rename(columns={0:"count", "index": "num_missing_features"}).sort_values(by = ["num_missing_features"])

    if plot:
        sns.barplot(x = "num_missing_features", y = "count", data=clean_df, color="navy")
        plt.title("Count of the number of missing features for all samples")
    
    else:
        return clean_df



def drop_missing_feature_threshold(df: pd.DataFrame, n_missing: int, plot = False) -> pd.DataFrame:
    """
    Drops samples that have more than n_missing features.
    """

    df["num_missing_features"] = row_level_missing(df, add_column=True)

    df_subset = df.loc[df["num_missing_features"] <= n_missing]
    print(f"Dropping samples with more than {n_missing} features missing...")
    print(f"Dropped {df.shape[0] - df_subset.shape[0]} samples")
    print(f"{df_subset.shape[0]} samples left")

    df_subset.drop(columns=["num_missing_features"],inplace=True)

    if plot:
        sns.boxplot(x="num_missing_features", y = "max_grade", color = "navy", data = df_subset)
        plt.title("Effect of number of missing features on grade")
    
    else:
        return df_subset

def knn_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses k-Nearest Neighbors to impute missing values.
    """

    imputer = KNNImputer(n_neighbors=3)
    
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

    return df_imputed

def write_imputed_data(df: pd.DataFrame, path: str = os.path.join("data", "processed")) -> None:
    """
    Writes the imputed data csv to the processed folder.
    """

    if not os.path.exists(path):
        print(f"Creating {path} directory...")
        os.makedirs(path)

    imputed_path = os.path.join(path, "imputed_data.csv")

    if not os.path.exists(imputed_path):
        df.to_csv(imputed_path)
        print(f"Imputed dataset saved to {imputed_path}")
    else:
        print(f"{imputed_path} already exists!")
    
def main():
    data_path = os.path.join("data", "processed")
    raw = pd.read_csv(os.path.join(data_path, "max_boulder_grade_users.csv")).iloc[:,1:]

    no_extremes_df = mask_extreme_values(raw)
    drop_missing_df = drop_missing_feature_threshold(no_extremes_df, 2)
    imputed_df = knn_imputer(drop_missing_df)

    write_imputed_data(imputed_df)

if __name__ == '__main__':
    main()


