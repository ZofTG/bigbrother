"""test saved data"""


#! IMPORTS


from os.path import dirname, join
from numpy import load


#! MAIN


if __name__ == "__main__":
    file = join(dirname(__file__), "webcam_data_10hz.npz")
    data = dict(load(file))
    print("\n".join(["TIMESTAMPS:"] + list(data.keys())))
