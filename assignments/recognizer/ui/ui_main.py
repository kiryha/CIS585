# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Recognizer(object):
    def setupUi(self, Recognizer):
        if not Recognizer.objectName():
            Recognizer.setObjectName(u"Recognizer")
        Recognizer.resize(294, 674)
        self.centralwidget = QWidget(Recognizer)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.label = QLabel(self.splitter)
        self.label.setObjectName(u"label")
        self.splitter.addWidget(self.label)
        self.linAlfa = QLineEdit(self.splitter)
        self.linAlfa.setObjectName(u"linAlfa")
        self.linAlfa.setMaximumSize(QSize(150, 16777215))
        self.splitter.addWidget(self.linAlfa)

        self.verticalLayout.addWidget(self.splitter)

        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.label_2 = QLabel(self.splitter_2)
        self.label_2.setObjectName(u"label_2")
        self.splitter_2.addWidget(self.label_2)
        self.linIterations = QLineEdit(self.splitter_2)
        self.linIterations.setObjectName(u"linIterations")
        self.linIterations.setMaximumSize(QSize(150, 16777215))
        self.splitter_2.addWidget(self.linIterations)

        self.verticalLayout.addWidget(self.splitter_2)

        self.splitter_6 = QSplitter(self.centralwidget)
        self.splitter_6.setObjectName(u"splitter_6")
        self.splitter_6.setOrientation(Qt.Horizontal)
        self.btnTrainSource = QPushButton(self.splitter_6)
        self.btnTrainSource.setObjectName(u"btnTrainSource")
        self.btnTrainSource.setMinimumSize(QSize(0, 35))
        self.splitter_6.addWidget(self.btnTrainSource)
        self.btnTrainExtended = QPushButton(self.splitter_6)
        self.btnTrainExtended.setObjectName(u"btnTrainExtended")
        self.btnTrainExtended.setMinimumSize(QSize(0, 35))
        self.splitter_6.addWidget(self.btnTrainExtended)

        self.verticalLayout.addWidget(self.splitter_6)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.btnExtendData = QPushButton(self.centralwidget)
        self.btnExtendData.setObjectName(u"btnExtendData")
        self.btnExtendData.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.btnExtendData)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.layImages = QVBoxLayout()
        self.layImages.setObjectName(u"layImages")

        self.verticalLayout.addLayout(self.layImages)

        self.splitter_4 = QSplitter(self.centralwidget)
        self.splitter_4.setObjectName(u"splitter_4")
        self.splitter_4.setOrientation(Qt.Horizontal)
        self.btnToSourceModel = QPushButton(self.splitter_4)
        self.btnToSourceModel.setObjectName(u"btnToSourceModel")
        self.splitter_4.addWidget(self.btnToSourceModel)
        self.btnToExtendedModel = QPushButton(self.splitter_4)
        self.btnToExtendedModel.setObjectName(u"btnToExtendedModel")
        self.splitter_4.addWidget(self.btnToExtendedModel)

        self.verticalLayout.addWidget(self.splitter_4)

        self.splitter_5 = QSplitter(self.centralwidget)
        self.splitter_5.setObjectName(u"splitter_5")
        self.splitter_5.setOrientation(Qt.Horizontal)
        self.labModel = QLabel(self.splitter_5)
        self.labModel.setObjectName(u"labModel")
        self.splitter_5.addWidget(self.labModel)
        self.linModel = QLineEdit(self.splitter_5)
        self.linModel.setObjectName(u"linModel")
        self.linModel.setEnabled(False)
        self.splitter_5.addWidget(self.linModel)

        self.verticalLayout.addWidget(self.splitter_5)

        self.btnLoadImage = QPushButton(self.centralwidget)
        self.btnLoadImage.setObjectName(u"btnLoadImage")
        self.btnLoadImage.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.btnLoadImage)

        self.splitter_3 = QSplitter(self.centralwidget)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Horizontal)
        self.label_3 = QLabel(self.splitter_3)
        self.label_3.setObjectName(u"label_3")
        self.splitter_3.addWidget(self.label_3)
        self.linIndex = QLineEdit(self.splitter_3)
        self.linIndex.setObjectName(u"linIndex")
        self.linIndex.setMaximumSize(QSize(150, 16777215))
        self.splitter_3.addWidget(self.linIndex)

        self.verticalLayout.addWidget(self.splitter_3)

        self.btnRecognize = QPushButton(self.centralwidget)
        self.btnRecognize.setObjectName(u"btnRecognize")
        self.btnRecognize.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.btnRecognize)

        self.btnDisplay = QPushButton(self.centralwidget)
        self.btnDisplay.setObjectName(u"btnDisplay")
        self.btnDisplay.setMinimumSize(QSize(0, 35))

        self.verticalLayout.addWidget(self.btnDisplay)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        Recognizer.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Recognizer)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 294, 21))
        Recognizer.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Recognizer)
        self.statusbar.setObjectName(u"statusbar")
        Recognizer.setStatusBar(self.statusbar)

        self.retranslateUi(Recognizer)

        QMetaObject.connectSlotsByName(Recognizer)
    # setupUi

    def retranslateUi(self, Recognizer):
        Recognizer.setWindowTitle(QCoreApplication.translate("Recognizer", u"Recognizer", None))
        self.label.setText(QCoreApplication.translate("Recognizer", u"Alfa: ", None))
        self.linAlfa.setText(QCoreApplication.translate("Recognizer", u"0.1", None))
        self.label_2.setText(QCoreApplication.translate("Recognizer", u"Iterations: ", None))
        self.linIterations.setText(QCoreApplication.translate("Recognizer", u"500", None))
        self.btnTrainSource.setText(QCoreApplication.translate("Recognizer", u"Train Source Model", None))
        self.btnTrainExtended.setText(QCoreApplication.translate("Recognizer", u"Train Extended Model", None))
        self.btnExtendData.setText(QCoreApplication.translate("Recognizer", u"Extend Source Data", None))
        self.btnToSourceModel.setText(QCoreApplication.translate("Recognizer", u"Source Model", None))
        self.btnToExtendedModel.setText(QCoreApplication.translate("Recognizer", u"Extended Model", None))
        self.labModel.setText(QCoreApplication.translate("Recognizer", u"Current Model:", None))
        self.btnLoadImage.setText(QCoreApplication.translate("Recognizer", u"Load Custom Image", None))
        self.label_3.setText(QCoreApplication.translate("Recognizer", u"MNIST Row Number", None))
        self.linIndex.setText(QCoreApplication.translate("Recognizer", u"1", None))
        self.btnRecognize.setText(QCoreApplication.translate("Recognizer", u"Recognize Test or Custom Image", None))
        self.btnDisplay.setText(QCoreApplication.translate("Recognizer", u"Display train.csv by MNIST Row Number", None))
    # retranslateUi

