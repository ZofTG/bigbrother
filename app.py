"""MULTICAMREADER APPLICATION"""


#! IMPORTS

import sys

from PyQt5 import QtWidgets

from src import MultiCameraReader


#! MAIN


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MultiCameraReader()
    mainwindow.show()
    status = app.exec()
    sys.exit(status)
