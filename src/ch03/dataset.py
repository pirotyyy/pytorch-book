import numpy as np


def load_data():
    data = np.array(
        [[166, 58.7], [176.0, 75.7], [171.0, 62.1], [173.0, 70.4], [169.0, 60.1]]
    )

    return (data[:, 0], data[:, 1])
