import sqlite3
import pandas as pd
import os

def db_connect(path: str = os.path.join("data", "raw", "8anu.sqlite")):
    """
    Creates a cursor to connect to the 
    """
    if not os.path.exists(path):
        print(f"Path to the Sqlite db {path} not found!")

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


def write_data(path: str = os.path.join("data", "processed")) -> None:
    """
    Writes two csvs to the data directory
    """

    if not os.path.exists(path):
        print(f"Creating {path} directory...")
        os.makedirs(path)

    max_boulder_path = os.path.join(path, "max_boulder_grade_users.csv")
    grade_path = os.path.join(path, "grade_conversion.csv")

    if not os.path.exists(max_boulder_path):
        print(f"Extracting user data to {max_boulder_path} ...")
        max_grade = extract_max_grade_data()
        max_grade.to_csv(max_boulder_path)
    else:
        print(f"{max_boulder_path} already exists!")

    if not os.path.exists(grade_path):
        print(f"Extracting grade conversion to {grade_path} ...")
        grades = get_grade_conversion_table()
        grades.to_csv(grade_path)
    else:
        print(f"{grade_path} already exists!")

def main():
    write_data()

if __name__ == '__main__':
    main()
