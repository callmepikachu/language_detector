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
    加载 Tatoeba 数据，并按指定语言等量采样。

    参数:
    path (str): Tatoeba 数据文件路径
    common_langs (List[str]): 要采样的语言列表，默认为 ['eng', 'fra', 'spa', 'deu', 'cmn']
    samples_per_lang (int): 每种语言采样数量

    返回值:
    List[Tuple[str, str]]: 包含 (text, language) 的列表
    """
    if common_langs is None:
        common_langs = ['eng', 'fra', 'spa', 'deu']

    df = pd.read_csv(path, sep='\t', header=None, on_bad_lines='skip')
    df.columns = ['id', 'language', 'text']
    df = df[['language', 'text']].dropna()

    # 筛选并等量采样
    sampled_dfs = []
    for lang in common_langs:
        lang_df = df[df['language'] == lang].sample(n=samples_per_lang, replace=True, random_state=42)
        sampled_dfs.append(lang_df)

    result_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    return [(row['text'], row['language']) for _, row in result_df.iterrows()]



def detect_tatoeba_data(data: List[Tuple[str, str]]):
    """
    分析给定数据集中各语言的分布情况。

    参数:
    data (List[Tuple[str, str]]): 包含 (text, language) 的列表

    返回值:
    dict: 包含总句数、语言计数、占比和排序后的语言列表
    """
    lang_counter = Counter(lang for _, lang in data)
    total = len(data)

    return {
        'total': total,
        'language_counts': dict(lang_counter),
        'language_proportions': {lang: count / total for lang, count in lang_counter.items()},
        'top_languages': sorted(lang_counter.items(), key=lambda x: x[1], reverse=True)
    }
