from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
import sys
from WindowUI import WindowUI
import os
import glob
from PyQt5 import QtGui,QtCore


import numpy as np

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     ui = WindowUI()
#     ui.show()
#     sys.exit(app.exec_())


if __name__ == '__main__':
    print(np.loadtxt('/home/loc/face_psych/checkpoints/label2city/loss_log.txt', delimiter=',', dtype=int))
    # print(a)
    # print(b)