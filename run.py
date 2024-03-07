"""app"""

#! IMPORTS


import sys
from PyQt5 import QtWidgets
from src import IRCam


#! MAIN


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = IRCam()
    mainwindow.show()
    status = app.exec()
    sys.exit(status)
