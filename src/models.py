from enum import Enum


class ImputationMethod(Enum):
    MEAN = "mean"
    KNN = "KNN (k-nearest neighbors)"
    RF = "RF (Random Forest)"
    MF = "MF (Multiple Factors)"
