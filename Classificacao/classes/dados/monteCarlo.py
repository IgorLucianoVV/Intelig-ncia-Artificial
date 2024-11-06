import numpy as np

class MonteCarlo():
    def __init__(self) -> None:
        pass
    @staticmethod
    def partition(x: np.ndarray, y: np.ndarray):
        n = x.shape[0]
        # inidice de partição
        part_n = int((n * 80)/100)

        # lista de indics aleitos
        index_list = np.random.permutation(np.arange(n))
        
        x_train, y_train = x[index_list[:part_n], :],y[index_list[:part_n], :]
        x_test, y_test = x[index_list[part_n:], :], y[index_list[part_n:], :]
        return (x_train, y_train, x_test, y_test)

    @staticmethod
    def partition2(x: np.ndarray, y: np.ndarray):
        n = x.shape[1]
        # inidice de partição
        part_n = int((n * 80)/100)

        # lista de indics aleitos
        index_list = np.random.permutation(np.arange(n))
        
        x_train, y_train = x[:, index_list[:part_n]],y[:, index_list[:part_n]]
        x_test, y_test = x[:, index_list[part_n:]], y[:, index_list[part_n:]]
        return (x_train, y_train, x_test, y_test)