"""MULTICAMREADER APPLICATION"""


#! IMPORTS

import sys

from PyQt5 import QtWidgets

from src import BigBrother


#! MAIN


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = BigBrother()
    mainwindow.show()
    status = app.exec()
    sys.exit(status)
