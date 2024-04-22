"""
Codes for running the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import warnings
import yaml
import tensorflow as tf
from SimuLine.Preprocessing.UnbiasedEmbedding.src.trainer import Trainer


def main():
    # possible_model_names = ['bpr', 'ubpr', 'wmf', 'relmf']
    # Run#1: 'ubpr' + use_pretrain  -- Ours
    # Run#2: 'bpr' +  use_pretrain
    # Run#3: 'bpr' + ~use_pretrain
    model_name = 'ubpr'  
    run_sims = 1
    data = 'Adressa'
    use_pretrain = True
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")

    config = yaml.safe_load(open('./SimuLine/Preprocessing/UnbiasedEmbedding/conf/config.yaml', 'rb'))
    trainer = Trainer(
        data=data,
        batch_size=config['batch_size'],
        max_iters=config['max_iters'],
        eta=config['eta'],
        model_name=model_name,
        use_pretrain=use_pretrain
    )
    trainer.run(num_sims=run_sims)

    print('\n', '=' * 25, '\n')
    print(f'Finished Running {model_name}!')
    print('\n', '=' * 25, '\n')
