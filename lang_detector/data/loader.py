import pandas as pd
from pydantic import BaseModel
from collections import Counter
from typing import List, Tuple, Optional


class DatasetSample(BaseModel):
    language: str
    text: str

def load_tatoeba_data(
    path: str = "data/sentences.csv",
    common_langs: Optional[List[str]] = None,
    samples_per_lang: int = 1000
) -> List[Tuple[str, str]]:
    """
    load Tatoeba dataset and sample it 。

    :param:
    path (str): Tatoeba data file path
    common_langs (List[str])
    samples_per_lang (int): how many sample per language

    :return:
    List[Tuple[str, str]]
    """
    if common_langs is None:
        common_langs = ['eng', 'fra', 'spa', 'deu', 'rus']

    df = pd.read_csv(path, sep='\t', header=None, on_bad_lines='skip')
    df.columns = ['id', 'language', 'text']
    df = df[['language', 'text']].dropna()

    sampled_dfs = []
    for lang in common_langs:
        lang_df = df[df['language'] == lang].sample(n=samples_per_lang, replace=True, random_state=42)
        sampled_dfs.append(lang_df)

    result_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    return [(row['text'], row['language']) for _, row in result_df.iterrows()]



def detect_tatoeba_data(data: List[Tuple[str, str]]):
    """
    detect diffenrent lang in data。

    :param:
    data (List[Tuple[str, str]]):

    :return:
    dict: Contains a list of total sentences, language count, percentage, and sorted languages
    """
    lang_counter = Counter(lang for _, lang in data)
    total = len(data)

    return {
        'total': total,
        'language_counts': dict(lang_counter),
        'language_proportions': {lang: count / total for lang, count in lang_counter.items()},
        'top_languages': sorted(lang_counter.items(), key=lambda x: x[1], reverse=True)
    }
