import os

import pandas as pd
from datasets import load_dataset


def get_dataset() -> pd.DataFrame:
    """
    Load the AG News dataset from Hugging Face and prepare it for use.
    Returns:
        pd.DataFrame: A DataFrame containing the text and labels of the AG News dataset.
    """

    print('Loading the AG News dataset from HuggingFace...')
    hf_dataset = load_dataset('ag_news', split='train')
    df = hf_dataset.to_pandas()

    # Data preparation
    class_names = [
        'World',
        'Sports',
        'Business',
        'Sci/Tech',
    ]  # 0 -> World, 1 -> Sports, 2 -> Business, 3 -> Sci/Tech
    df['label'] = df['label'].apply(lambda x: class_names[x])

    print('- Dataset loaded and prepared successfully.')
    return df[['text', 'label']]


if __name__ == '__main__':
    get_dataset().to_parquet(
        os.path.join(os.path.dirname(__file__), '..', 'resources', 'dataset.parquet'),
        index=False,
    )
    print('- Dataset saved to resources/dataset.parquet')
