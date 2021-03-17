import json
import logging
import numpy as np
import os
import pandas as pd
import psycopg2
import sys

from datetime import datetime
from google.cloud import storage
from io import StringIO
from query_create_table import query_create_table
from time import sleep


# For testing purposes, set pandas output to display all rows / columns
debug_mode = True
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set up logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def connect_to_bucket(service_account: str) -> storage.bucket.Bucket:
    # Connect to and return the case study Google Cloud Storage bucket (fst-python-case-study-test)
    PATH = os.path.join(os.getcwd(), service_account)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PATH
    storage_client = storage.Client(PATH)

    return storage_client.get_bucket('fst-python-case-study-test')


def connect_to_database(credentials_file: str) -> psycopg2.connect:
    # Connect to the PostgreSQL database server
    with open(credentials_file) as f:
        credentials = json.load(f)

    try:
        conn = psycopg2.connect(**credentials)
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)
        sys.exit(1)
    return conn


def execute_query(conn: psycopg2.connect, query: str) -> None:
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Unable to execute query. Error: {error}")
        conn.rollback()
        cursor.close()

    cursor.close()


def check_new_files(bucket: storage.bucket.Bucket) -> np.ndarray:
    # Retrieve local list of files previously loaded into the database and compare to the files currently in the bucket
    loaded_files = pd.read_csv('loaded_files.csv')['file name']
    bucket_files = [filename.name for filename in list(bucket.list_blobs())]
    new_files = np.setdiff1d(bucket_files, loaded_files.tolist(), assume_unique=False)

    return new_files


def load_file(file_name: str) -> pd.DataFrame:
    """
    Function to load in data from csv and return a pandas dataframe. Currently can handle .csv and .csv.gz, but could be
    extended to handle further file types.
    Converts the time column from str -> datetime objects and removes any commas in the data (as these can disrupt the
    csv creation when importing the data into the database in insert_data()).
    """
    file_location = 'gs://fst-python-case-study-test/' + file_name

    if file_name.endswith('.gz'):
        df = pd.read_csv(file_location, compression='gzip')
    elif file_name.endswith('.csv'):
        df = pd.read_csv(file_location)
    else:
        logging.error("File type not recognised: must be of type csv")
        sys.exit(1)

    df['Time'] = pd.to_datetime(df.Time.str[:], format="%Y-%m-%d-%H:%M:%S")
    df = df.replace(',', '-', regex=True)
    return df


def set_index(conn: psycopg2.connect, df: pd.DataFrame) -> None:
    """
    Adjust the index of the dataframe, such that it starts from the maximum entryID of the AdvertisementData table and
    auto-increments. This avoids duplicate key entry errors when inserting data into the database.
    """
    query = "SELECT MAX(entryid) FROM advertisementdata"
    index = pd.read_sql(query, conn)["max"][0]

    if index is not None:
        df.index = np.arange(index + 1, len(df) + index + 1)
    else:
        df.index = np.arange(1, len(df) + 1)


def insert_data(conn: psycopg2.connect, df: pd.DataFrame) -> None:
    set_index(conn, df)

    # Bulk insert dataframe into advertisementdata table
    buffer = StringIO()
    df.to_csv(buffer, index_label='id', header=False)
    buffer.seek(0)
    cursor = conn.cursor()

    try:
        cursor.copy_from(buffer, 'advertisementdata', sep=",")
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Error inserting data: {error}")
        conn.rollback()
        cursor.close()

    cursor.close()


def update_loaded_files_list(filename: str) -> None:
    """
    Once data has been successfully loaded into the table, append the file name to the locally stored list
    "loaded_files.csv"
    """
    file_added = pd.DataFrame({'file name': [filename]})
    file_added.to_csv('loaded_files.csv', mode='a', index=False, header=False)


def remove_duplicates(conn: psycopg2.connect) -> int:
    """
    Remove duplicates from the table once data has been added by assigning a running count to duplicate rows and then
    removing the corresponding rows with a count >= 2. A row's uniqueness is determined by the time of the event
    and the EventKeyPart.
    Return the total number of duplicates removed from the table.
    """
    query_count = '''SELECT COUNT(*) AS count FROM advertisementdata'''

    query_remove_duplicates = '''DELETE FROM advertisementdata 
                                WHERE entryid IN (SELECT entryid FROM(
                                SELECT *, ROW_NUMBER() OVER(
                                PARTITION BY time, EventKeyPart ORDER BY entryid) 
                                FROM advertisementdata)s 
                                WHERE ROW_NUMBER >=2);'''

    initial_number_of_records = pd.read_sql(query_count, conn).iloc[0]['count']
    execute_query(conn, query_remove_duplicates)
    final_number_of_records = pd.read_sql(query_count, conn).iloc[0]['count']

    return initial_number_of_records - final_number_of_records


# Analysis functions to generate reports

def get_records_per_day(conn: psycopg2.connect) -> pd.DataFrame:
    # Total number of records per day
    query = '''SELECT time::date AS date, COUNT(*) AS count FROM advertisementdata GROUP BY date'''
    return pd.read_sql(query, conn)


def get_records_per_hour(conn: psycopg2.connect) -> pd.DataFrame:
    # Total number of records per hour
    query = '''SELECT DATE_TRUNC('hour', time) AS datetime, COUNT(*) AS count FROM advertisementdata GROUP BY datetime ORDER BY datetime'''
    return pd.read_sql(query, conn)


def get_total_ebfr_per_day(conn: psycopg2.connect) -> pd.DataFrame:
    # Total estimated backfill revenue per day
    query = '''SELECT time::date AS date, sum(estimatedbackfillrevenue) AS total_estimated_backfill_revenue FROM advertisementdata GROUP BY date'''
    return pd.read_sql(query, conn)


def get_total_ebfr_per_hour(conn: psycopg2.connect) -> pd.DataFrame:
    # Total estimated backfill revenue per day
    query = '''SELECT DATE_TRUNC('hour', time) AS datetime, 
                sum(estimatedbackfillrevenue) AS total_estimated_backfill_revenue 
                FROM advertisementdata 
                GROUP BY datetime 
                ORDER BY datetime'''
    return pd.read_sql(query, conn)


# def get_total_records_per_buyer(conn: psycopg2.connect) -> pd.DataFrame:
    # Total records per buyer
#    query = '''SELECT buyer, COUNT(*) AS count FROM advertisementdata GROUP BY buyer ORDER BY buyer'''
#    return pd.read_sql(query, conn)


def get_total_ebfr_per_buyer(conn: psycopg2.connect) -> pd.DataFrame:
    # Total estimated backfill revenue per buyer with total number of records
    query = '''SELECT buyer, sum(estimatedbackfillrevenue) AS total_estimated_backfill_revenue, COUNT(*) AS count 
                FROM advertisementdata 
                GROUP BY buyer 
                ORDER BY buyer'''
    return pd.read_sql(query, conn)


def get_unique_device_categories_per_buyer(conn: psycopg2.connect) -> pd.DataFrame:
    # Unique device categories per buyer
    query = '''SELECT advertiser, devicecategory, COUNT(*) as count 
                FROM advertisementdata 
                GROUP BY advertiser, devicecategory 
                ORDER BY advertiser'''
    return pd.read_sql(query, conn)


def analysis(conn: psycopg2.connect) -> None:
    """
    Execute all of the above analysis functions and output the data to individual timestamped csv files saved in the
    directory 'reports'.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    get_records_per_day(conn).to_csv(f'reports/records_per_day_{timestamp}.csv')
    get_records_per_hour(conn).to_csv(f'reports/records_per_hour_{timestamp}.csv')
    get_total_ebfr_per_day(conn).to_csv(f'reports/total_ebfr_per_day_{timestamp}.csv')
    get_total_ebfr_per_hour(conn).to_csv(f'reports/total_ebfr_per_hour_{timestamp}.csv')
    #get_total_records_per_buyer(conn).to_csv(f'reports/total_records_per_buyer_{timestamp}.csv')
    get_total_ebfr_per_buyer(conn).to_csv(f'reports/total_ebfr_per_buyer_{timestamp}.csv')
    get_unique_device_categories_per_buyer(conn).to_csv(f'reports/unique_device_categories_per_buyer_{timestamp}.csv')


def main():
    # Establish connections to the database and storage bucket
    conn = connect_to_database("conn_credentials2.json")
    bucket = connect_to_bucket("gcp-casestudy-32e8e2a33790.json")

    if debug_mode:
        # If in debug mode, drop and re-create the table (for testing purposes)
        execute_query(conn, '''DROP TABLE advertisementdata''')
        execute_query(conn, query_create_table)

    while True:
        data_added = False
        duplicates_removed = 0

        logging.info("Checking for new files")

        for file in check_new_files(bucket):
            logging.info(f"Loading file {file}")
            df = load_file(file)
            logging.info(f"Inserting data into table: advertisementdata")
            insert_data(conn, df)
            update_loaded_files_list(file)
            logging.info(f"Successfully inserted {file} to database")
            data_added = True

        if data_added:
            logging.info("Removing duplicates from table: advertisementdata")
            duplicates_removed += remove_duplicates(conn)
            logging.info(f"Duplicates removed: {duplicates_removed}")
            logging.info(f"Analysing table: advertisementdata. Results saved to {os.getcwd()}/reports")
            analysis(conn)
            logging.info("Analysis complete")

        sleep(10)  # Sleep for 10 seconds before checking for new files in the storage bucket.


if __name__ == "__main__":
    main()
