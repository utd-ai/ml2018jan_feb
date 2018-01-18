from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import imp
import signal
import sys


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self):
        self.X_data, self.y_data = make_classification(
            n_samples=10000, n_features=20, 
            n_classes=2, n_informative=20, 
            n_redundant=0,
            random_state=42
        )
        self.applications = 0

    def check(self, script_path):
        try:
            # Минута на выполнение алгоритма
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(60)
            svm_impl = imp.load_source('svm_impl_{}'.format(self.applications), script_path)
            self.applications += 1
            # В модуле должна быть объявлена константа SVM_PARAMS_DICT, 
            # содержащая в себе значения параметров, необходимых для вашего алгоритма
            algo = svm_impl.MySVM(**svm_impl.SVM_PARAMS_DICT)
            return np.mean(cross_val_score(algo, self.X_data, self.y_data, cv=2, scoring='accuracy'))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            return None


if __name__ == '__main__':
    print(Checker().check(SCRIPT_DIR + '/svm_impl_example.py'))
