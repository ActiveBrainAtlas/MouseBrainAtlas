# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RectificationTool.ui'
#
# Created: Sun Aug  7 15:45:34 2016
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_RectificationGUI(object):
    def setupUi(self, RectificationGUI):
        RectificationGUI.setObjectName(_fromUtf8("RectificationGUI"))
        RectificationGUI.resize(1521, 1113)
        self.centralwidget = QtGui.QWidget(RectificationGUI)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.splitter_3 = QtGui.QSplitter(self.centralwidget)
        self.splitter_3.setOrientation(QtCore.Qt.Vertical)
        self.splitter_3.setObjectName(_fromUtf8("splitter_3"))
        self.splitter = QtGui.QSplitter(self.splitter_3)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.sagittal_gview = QtGui.QGraphicsView(self.splitter)
        self.sagittal_gview.setObjectName(_fromUtf8("sagittal_gview"))
        self.coronal_gview = QtGui.QGraphicsView(self.splitter)
        self.coronal_gview.setObjectName(_fromUtf8("coronal_gview"))
        self.splitter_2 = QtGui.QSplitter(self.splitter_3)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName(_fromUtf8("splitter_2"))
        self.horizontal_gview = QtGui.QGraphicsView(self.splitter_2)
        self.horizontal_gview.setObjectName(_fromUtf8("horizontal_gview"))
        self.layoutWidget = QtGui.QWidget(self.splitter_2)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 0, 4, 1, 1)
        self.label = QtGui.QLabel(self.layoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 1, 1, 1)
        self.slider_hxy = QtGui.QSlider(self.layoutWidget)
        self.slider_hxy.setMinimum(-100)
        self.slider_hxy.setMaximum(100)
        self.slider_hxy.setTracking(True)
        self.slider_hxy.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hxy.setInvertedAppearance(False)
        self.slider_hxy.setInvertedControls(False)
        self.slider_hxy.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slider_hxy.setObjectName(_fromUtf8("slider_hxy"))
        self.gridLayout.addWidget(self.slider_hxy, 1, 2, 1, 1)
        self.label_4 = QtGui.QLabel(self.layoutWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 1, 3, 1, 1)
        self.slider_hxz = QtGui.QSlider(self.layoutWidget)
        self.slider_hxz.setMinimum(-100)
        self.slider_hxz.setMaximum(100)
        self.slider_hxz.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hxz.setObjectName(_fromUtf8("slider_hxz"))
        self.gridLayout.addWidget(self.slider_hxz, 1, 4, 1, 1)
        self.label_2 = QtGui.QLabel(self.layoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 2, 1, 1, 1)
        self.slider_hyx = QtGui.QSlider(self.layoutWidget)
        self.slider_hyx.setMinimum(-100)
        self.slider_hyx.setMaximum(100)
        self.slider_hyx.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hyx.setObjectName(_fromUtf8("slider_hyx"))
        self.gridLayout.addWidget(self.slider_hyx, 2, 2, 1, 1)
        self.label_5 = QtGui.QLabel(self.layoutWidget)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 2, 3, 1, 1)
        self.slider_hyz = QtGui.QSlider(self.layoutWidget)
        self.slider_hyz.setMinimum(-100)
        self.slider_hyz.setMaximum(100)
        self.slider_hyz.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hyz.setObjectName(_fromUtf8("slider_hyz"))
        self.gridLayout.addWidget(self.slider_hyz, 2, 4, 1, 1)
        self.button_symmetric = QtGui.QPushButton(self.layoutWidget)
        self.button_symmetric.setObjectName(_fromUtf8("button_symmetric"))
        self.gridLayout.addWidget(self.button_symmetric, 2, 5, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 3, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.layoutWidget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 3, 1, 1, 1)
        self.slider_hzx = QtGui.QSlider(self.layoutWidget)
        self.slider_hzx.setMinimum(-100)
        self.slider_hzx.setMaximum(100)
        self.slider_hzx.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hzx.setObjectName(_fromUtf8("slider_hzx"))
        self.gridLayout.addWidget(self.slider_hzx, 3, 2, 1, 1)
        self.label_6 = QtGui.QLabel(self.layoutWidget)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 3, 3, 1, 1)
        self.slider_hzy = QtGui.QSlider(self.layoutWidget)
        self.slider_hzy.setMinimum(-100)
        self.slider_hzy.setMaximum(100)
        self.slider_hzy.setOrientation(QtCore.Qt.Horizontal)
        self.slider_hzy.setObjectName(_fromUtf8("slider_hzy"))
        self.gridLayout.addWidget(self.slider_hzy, 3, 4, 1, 1)
        self.button_midline = QtGui.QPushButton(self.layoutWidget)
        self.button_midline.setObjectName(_fromUtf8("button_midline"))
        self.gridLayout.addWidget(self.button_midline, 3, 5, 1, 1)
        self.button_done = QtGui.QPushButton(self.layoutWidget)
        self.button_done.setObjectName(_fromUtf8("button_done"))
        self.gridLayout.addWidget(self.button_done, 4, 5, 1, 1)
        self.button_cancel = QtGui.QPushButton(self.layoutWidget)
        self.button_cancel.setObjectName(_fromUtf8("button_cancel"))
        self.gridLayout.addWidget(self.button_cancel, 1, 5, 1, 1)
        self.gridLayout_2.addWidget(self.splitter_3, 0, 0, 1, 1)
        RectificationGUI.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(RectificationGUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1521, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        RectificationGUI.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(RectificationGUI)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        RectificationGUI.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(RectificationGUI)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        RectificationGUI.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(RectificationGUI)
        QtCore.QMetaObject.connectSlotsByName(RectificationGUI)

    def retranslateUi(self, RectificationGUI):
        RectificationGUI.setWindowTitle(_translate("RectificationGUI", "MainWindow", None))
        self.label.setText(_translate("RectificationGUI", "h_xy", None))
        self.label_4.setText(_translate("RectificationGUI", "h_xz", None))
        self.label_2.setText(_translate("RectificationGUI", "h_yx", None))
        self.label_5.setText(_translate("RectificationGUI", "h_yz", None))
        self.button_symmetric.setText(_translate("RectificationGUI", "Symmetric", None))
        self.label_3.setText(_translate("RectificationGUI", "h_zx", None))
        self.label_6.setText(_translate("RectificationGUI", "h_zy", None))
        self.button_midline.setText(_translate("RectificationGUI", "Midline", None))
        self.button_done.setText(_translate("RectificationGUI", "Complete", None))
        self.button_cancel.setText(_translate("RectificationGUI", "Cancel", None))
        self.toolBar.setWindowTitle(_translate("RectificationGUI", "toolBar", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    RectificationGUI = QtGui.QMainWindow()
    ui = Ui_RectificationGUI()
    ui.setupUi(RectificationGUI)
    RectificationGUI.show()
    sys.exit(app.exec_())

