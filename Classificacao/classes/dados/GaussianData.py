import numpy as np
from classes.dados.dados import DataHandler as dh

class GausianDataHandler(dh):
    def __init__(self, path) -> None:
        super().__init__(path)
        # os dados são organizados seguindo:
        # X ∈ R p×N; Y ∈ R C×N;
        self.gausianXY()


