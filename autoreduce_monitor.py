#!/usr/bin/env python


import sys
from PyQt4 import QtGui, QtCore
from time import strftime
import datetime
import os

status_file = '/SNS/MANDI/shared/autoreduce/autoreduce_status.txt'
last_run_file = '/SNS/MANDI/shared/autoreduce/last_run_processed.dat'

with open(last_run_file, 'r') as f:
    lastRun = int(f.readline())
with open(status_file, 'r') as f:
    status = str(f.readline())
    ipts_reading = int(f.readline())


class Example(QtGui.QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()

    def initUI(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.Time)
        self.timer.start(1000)

        self.lbl1 = QtGui.QLabel('Initializing MaNDi Autoreduction', self)
        self.lbl1.move(15, 10)
        self.lbl1.resize(500, 15)

        self.lbl2 = QtGui.QLabel(
             'Last Run Processed: {0}'.format(lastRun), self)
        self.lbl2.move(15, 40)
        self.lbl2.resize(500, 15)

        self.lbl3 = QtGui.QLabel('', self)
        self.lbl3.move(15, 70)
        self.lbl3.resize(500, 15)

        self.lbl4 = QtGui.QLabel(
             'Currently reading IPTS {}'.format(ipts_reading), self)
        self.lbl4.move(15, 100)
        self.lbl4.resize(500, 15)

        self.setGeometry(300, 300, 500, 150)
        self.setWindowTitle('MaNDi Autoreduction')
        self.show()

    def Time(self):
        if int(strftime("%S")) % 10 == 0:
            self.timer.setInterval(5000)
        else:
            self.timer.setInterval(5000)
        with open(status_file, 'r') as f:
            line = f.readline()
            ipts_reading = f.readline()
        self.lbl1.setText(line)

        statbuf = os.stat(status_file)
        datemod = datetime.datetime.fromtimestamp(statbuf.st_mtime)
        now = datetime.datetime.now()
        dt = now - datemod
        if 'Finished' in line:
            self.lbl2.setText("Finished at {}".format(str(datemod)))
            self.lbl3.setText(
                "Time since last finished analysis: {}".format(str(dt)))
        else:
            self.lbl2.setText(
                "Started processing at {}".format(str(datemod)))
            self.lbl3.setText(
                "Time since started analysis: {}".format(str(dt)))
        self.lbl4.setText(
                "Currently scanning IPTS {}".format(str(ipts_reading)))


def main():
    app = QtGui.QApplication(sys.argv)
    ex = Example()  # noqa: F841
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
"""
import sys, os, datetime
from PyQt4 import QtGui, QtCore
from time import strftime

status_file = '/SNS/MANDI/shared/autoreduce/autoreduce_status.txt'

class Main(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MaNDi autoreduction monitor')
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.Time)
        self.timer.start(1000)
        self.lcd1 = QtGui.QLineEdit()
        self.lcd1.setText('MANDI_AutoReduction')
        self.lcd1.move(20,20)
        self.lcd1.resize(280,40)

        self.setCentralWidget(self.lcd1)
        self.setGeometry(0,0,320,150)

    def Time(self):
        if int(strftime("%S")) % 10 == 0:
            self.timer.setInterval(5000)
        else:
            self.timer.setInterval(1000)
        with open(status_file, 'r') as f:
            line = f.readline()
        statbuf = os.stat(status_file)
        mod_time = str(datetime.datetime.fromtimestamp(statbuf.st_mtime))
        self.lcd1.setText(line)

def main():
    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
"""
