import numpy as np
import random

def set_seed(seed):
    

    np.random.seed(seed)
    random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass