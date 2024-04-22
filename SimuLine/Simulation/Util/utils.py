import os
import numpy as np
from recbole.utils import set_color

def dict_to_str(d, title):
    args_info = set_color(f'{title}\n', 'pink')
    args_info += '\n'.join([(set_color("{}", 'cyan') + " =" + set_color(" {}", 'yellow')).format(arg, value)
                            for arg, value in d.items()])
    args_info += '\n\n\n'
    return args_info

def np_to_str(np_vector):
    np_vector = [str(x) for x in np_vector.tolist()]
    return '\t'.join(np_vector)

def str_to_np(str_vector):
    str_vector = str_vector.split('\t')
    str_vector = [float(x) for x in str_vector]
    return np.array(str_vector)

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
