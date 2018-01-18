from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
import signal
import os
import json
import sys


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self):
        self.X, self.y = load_breast_cancer(return_X_y=True)


    def check(self, params_path):
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)

            # Минута на выполнение алгоритма
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(60)
            # оценивается качество по кроссвалидации у логистической регрессии с переданными параметрами
            score = np.mean(cross_val_score(
                LogisticRegression(**params), 
                self.X, 
                self.y,
                scoring='accuracy', 
                cv=3
            ))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            score = None
        
        return score


if __name__ == '__main__':
    print(Checker().check(SCRIPT_DIR + '/classification_params_example.json'))
