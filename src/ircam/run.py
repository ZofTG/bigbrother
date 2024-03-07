"""MULTICAMREADER APPLICATION"""

#! IMPORTS

import sys

from PyQt5 import QtWidgets

from .widgets import IRCam


__all__ = ["run"]

#! FUNCTIONS


def run():
    """run the IRCam app"""
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = IRCam()
    mainwindow.show()
    status = app.exec()
    sys.exit(status)


#! MAIN


if __name__ == "__main__":
    run()
