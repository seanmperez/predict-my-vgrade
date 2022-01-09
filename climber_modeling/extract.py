import sqlite3
import pandas as pd
import numpy as np
from math import floor, ceil
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def db_connect(path: str = os.path.join("data", "raw", "8anu.sqlite")):
    """
    Creates a cursor to connect to the sqlite db.
    """
    
    SQL_DB_URL = "https://www.kaggle.com/dcohen21/8anu-climbing-logbook"

    if not os.path.exists(path):
        print(f"Path to the Sqlite db {path} not found!")
        print(f"Consider downloading from {SQL_DB_URL} and saving to {path}")

    return sqlite3.connect(path)

def query_to_df(query: str) -> pd.DataFrame:
    """
    Takes a sql query, and a connection.
    Returns a dataframe.
    """
    con = db_connect()
    df = pd.read_sql(query, con)
    return df 

def extract_max_grade_data() -> pd.DataFrame:
    """
    Extracts the max bouldering grade, and other physicsal for each user.
    """

    max_grade_df = query_to_df(
        """
        SELECT u.height, u.weight, MAX(a.grade_id) AS max_grade, (2017 - started) AS years_climbing, (2017 - CAST(birth AS INT)) AS age, u.sex
        FROM user u
        LEFT JOIN ascent a
            ON u.id = a.user_id
        WHERE a.climb_type == 1
        GROUP BY u.id
        """
    )

    return max_grade_df

def get_grade_conversion_table() -> pd.DataFrame:
    """
    Extracts the grade table, including the convenient grade id, french, and US grades.
    """

    grade_conversion_df =  query_to_df(
        """
        SELECT id AS grade_id, usa_boulders, fra_boulders
        FROM grade
        ORDER BY id DESC
        """
    )

    return grade_conversion_df

def splitvgrade(x):
    """
    Returns V-grade floats as V-grade splits.
    """
    if x.is_integer():
        return f"V{int(x)}"
    else:
        return f"V{floor(x)}/V{ceil(x)}"

def fix_v_grades(df):
    """
    Corrects V-grade conversions
    """
    df_copy = df.copy()

    # Set max grade to V17 / 9A
    df_copy.loc[(df['grade_id'] > 75), ["usa_boulders", "fra_boulders"]] = ["V17", "9A"]
    
    # Set minimum grade to VB / 3
    df_copy.loc[(df['grade_id'] < 12), ["usa_boulders", "fra_boulders"]] = ["VB", "3"]

    # Remove grades with a slash
    df_copy.loc[df['usa_boulders'].str.contains("/", na=False), ["usa_boulders"]] = np.nan

    # Remove V
    df_copy["usa_boulders"] = df_copy["usa_boulders"].str.replace("V", "")

    # Remove - sign at end of string
    df_copy["usa_boulders"] = df_copy["usa_boulders"].str.replace("-", "")

    # Replace B with -1 
    df_copy["usa_boulders"] = df_copy["usa_boulders"].str.replace("B", "-1")

    # Change grade to int
    df_copy["usa_boulders"] = df_copy["usa_boulders"].replace("", np.nan).astype(float)

    # Make foward and back fill columns
    df_copy["ffill"] = df_copy["usa_boulders"].fillna(method = "ffill")
    df_copy["bfill"] = df_copy["usa_boulders"].fillna(method = "bfill")

    # Set grade as averages of fill methods
    df_copy["usa_boulders"] = df_copy[["ffill","bfill"]].mean(axis=1)
    
    # Make V-grade splits rather than decimals
    df_copy["usa_boulders"] = df_copy["usa_boulders"].apply(splitvgrade)

    # Replace V-1 with VB
    df_copy["usa_boulders"] = df_copy["usa_boulders"].str.replace("-1", "B")
    
    return df_copy[["grade_id", "usa_boulders", "fra_boulders"]]

def write_data(path: str = os.path.join("data", "processed")) -> None:
    """
    Writes two csvs to the data directory:
    1. grade_conversion.csv - A table to convert the 8anu grade system to V-grade and french grades.
    2. max_boulder_grade_users.csv - The raw climbing data extracted from SQL.
    """

    if not os.path.exists(path):
        print(f"Creating {path} directory...")
        os.makedirs(path)

    max_boulder_path = os.path.join(path, "max_boulder_grade_users.csv")
    grade_path = os.path.join(path, "grade_conversion.csv")

    if not os.path.exists(max_boulder_path):
        print(f"Extracting user data to {max_boulder_path} ...")
        max_grade = extract_max_grade_data()
        max_grade.to_csv(max_boulder_path, index = False)
    else:
        print(f"{max_boulder_path} already exists!")

    if not os.path.exists(grade_path):
        print(f"Extracting grade conversion to {grade_path} ...")
        raw_grades = get_grade_conversion_table()
        grades = fix_v_grades(raw_grades)
        grades.to_csv(grade_path, index = False)
    else:
        print(f"{grade_path} already exists!")

def main():
    write_data()

if __name__ == '__main__':
    main()
