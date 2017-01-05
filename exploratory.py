import pandas as pd
import numpy as np


random_weights = {
        "horse": lambda: max(0, np.random.normal(5, 2, 1)[0]),
        "ball": lambda: max(0, 1 + np.random.normal(1, 0.3, 1)[0]),
        "bike": lambda: max(0, np.random.normal(20, 10, 1)[0]),
        "train": lambda: max(0, np.random.normal(10, 5, 1)[0]),
        "coal": lambda: 47 * np.random.beta(0.5, 0.5, 1)[0],
        "book": lambda: np.random.chisquare(2, 1)[0],
        "doll": lambda: np.random.gamma(5, 1, 1)[0],
        "blocks": lambda: np.random.triangular(5, 10, 20, 1)[0],
        "gloves": lambda: 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
    }


def random_weight_list(gift_type, sample_size, gifts_per_sample):
    weight_list = list()
    for i in xrange(sample_size):
        weight_list.append(sum([random_weights[gift_type]() * gifts_per_sample]))
    return weight_list


df = pd.DataFrame({'ball_23': random_weight_list('ball', 500, 23),
                  'ball_24': random_weight_list('ball', 500, 24)})