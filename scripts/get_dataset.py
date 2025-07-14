import os

import pandas as pd

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources')


def get_dataset() -> pd.DataFrame:
    """
    Load the Amazon Sales Dataset from a CSV file.
    The dataset is from the following Kaggle repository: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
    The dataset is expected to be in the 'resources' directory as 'amazon.csv'.
    Returns:
        pd.DataFrame: A DataFrame containing the text and labels of the Amazon Sales
        Dataset.
    """

    print('Reading the Amazon CSV...')
    df = pd.read_csv(
        os.path.join(RESOURCES_DIR, 'amazon.csv'),
        usecols=['product_name', 'category'],
        dtype={'product_name': str, 'category': str},
    )

    # Data preparation
    df = df.drop_duplicates(subset=['product_name'])[['product_name', 'category']]
    df.rename(columns={'product_name': 'text'}, inplace=True)
    df['category'] = df['category'].apply(
        lambda x: x.split('|')[0] if isinstance(x, str) else x
    )

    print('- Dataset loaded and prepared successfully.')
    return df


if __name__ == '__main__':
    get_dataset().to_parquet(
        os.path.join(os.path.dirname(__file__), '..', 'resources', 'dataset.parquet'),
        index=False,
    )
    print('- Dataset saved to resources/dataset.parquet')
