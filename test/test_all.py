"""test saved data"""

#! IMPORTS


from os.path import dirname, join
from numpy import load
import subprocess


#! MAIN


def test_data_reading():
    try:
        file = join(dirname(__file__), "webcam_data_10hz.npz")
        data = dict(load(file))
    except Exception:
        raise RuntimeError("Error in data reading")


def test_app_launch():
    try:
        subprocess.run(["python", "src/gui.py"])
    except Exception:
        raise RuntimeError("Error in launching the app")
