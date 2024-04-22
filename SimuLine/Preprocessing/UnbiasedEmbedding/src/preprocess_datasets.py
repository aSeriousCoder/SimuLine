"""
Codes for preprocessing the real-world datasets
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import warnings
from SimuLine.Preprocessing.UnbiasedEmbedding.src.preprocess.preprocessor import preprocess_dataset


def main():
    warnings.filterwarnings("ignore")
    data = 'Adressa'

    preprocess_dataset(data=data)

    print('\n', '=' * 25, '\n')
    print(f'Finished Preprocessing {data}!')
    print('\n', '=' * 25, '\n')
