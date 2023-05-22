import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
import tensorflow as tf
import os
import glob
import pandas as pd
import gensim
from pyevmasm import instruction_tables, disassemble_hex, disassemble_all, assemble_hex
import binascii
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
import re
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
import xgboost
from scipy.sparse import hstack, coo_matrix, csr_matrix
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
import joblib

# Disassemble bytecode to opcode func
def preprocess_text(text):
    new_text = text.replace("0x", "")
    return new_text


def clean_text(text):
    new_a = text.replace("\n", " ")
    return new_a


def BytecodeToOpcode(bytecode):
    instruction_table = instruction_tables["byzantium"]
    instruction_table[20]
    instruction_table["EQ"]
    instrs = list(disassemble_all(binascii.unhexlify(bytecode)))
    instrs.insert(1, instruction_table["JUMPI"])
    a = assemble_hex(instrs)
    opcode = disassemble_hex(a)
    return opcode


def getFiles(filepath):
    all_files = []
    for root, dirs, files in os.walk(filepath):
        files = glob.glob(os.path.join(root, "*.json"))
        for f in files:
            all_files.append(os.path.abspath(f))
    return all_files


def listToString(s):

    # initialize an empty string
    str1 = " "

    # traverse in the string
    for ele in s:
        str1 += " " + ele

    # return string
    return str1


def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos="v")


def dummy(doc):
    return doc


def preprocess_opcode(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        token = re.sub(r"[^\x00-\x7f]", r"", token)
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def main(tx_paths, tx_list, label, output, path_images):
    file_results = open(output, "w")
    file_results.write(
        "Scores of the performance evaluation are: Accuracy, Precision, Recall, F1-score\n"
    )

    for tx in tx_paths:
        with open(tx) as doc:
            exp = json.load(doc)
            tx_list.append(exp)
        parts = tf.strings.split(tx, os.path.sep)
        if parts[-2] == "Delegatecall":
            label.append("Delegatecall")
        elif parts[-2] == "DoS":
            label.append("DoS")
        elif parts[-2] == "FDV":
            label.append("FDV")
        elif parts[-2] == "FoT":
            label.append("FoT")
        elif parts[-2] == "Normal":
            label.append("Normal")
        elif parts[-2] == "OaU":
            label.append("OaU")

    # load data to DataFrame
    df = pd.DataFrame(tx_list)
    df["Label"] = label

    # convert bytecode to opcode
    pre_text = df["input"].apply(preprocess_text)
    text = pre_text.apply(BytecodeToOpcode)
    opcode = text.apply(clean_text)
    preprocessed_opcode = opcode.apply(preprocess_opcode)

    # convert preprocessed opcode to vector
    cv = TfidfVectorizer(
        tokenizer=dummy,
        preprocessor=dummy,
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(2, 3),
        max_features=30000,
    )
    Data_noOp = cv.fit_transform(preprocessed_opcode)
    y = df.Label
    # convert opcode with operands to vector
    cv2 = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(2, 3),
        max_features=30000,
    )
    Data_operand = cv2.fit_transform(opcode)

    # SVM with no operands opcode
    # x_train, x_test, y_train, y_test = train_test_split(
    #    Data, y, test_size=0.2, random_state=0
    # )
    # classifier = SVC(kernel="linear", random_state=0)
    # classifier.fit(x_train, y_train)
    # pred = classifier.predict(x_test)
    # cm = confusion_matrix(y_test, pred)
    # report = classification_report(y_test, pred, digits=5)
    # disp = ConfusionMatrixDisplay(cm)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.savefig(path_images + "/output.png")
    # file_results.write("Result of SVM with no operands opcode\n")
    # file_results.write(str(report))

    # SVM with operands opcode
    # x_train_opr, x_test_opr, y_train_opr, y_test_opr = train_test_split(
    #    Data_operand, y, test_size=0.2, random_state=0
    # )
    # classifier.fit(x_train_opr, y_train_opr)
    # pred1 = classifier.predict(x_test_opr)
    # cm1 = confusion_matrix(y_test_opr, pred1)
    # report1 = classification_report(y_test_opr, pred1, digits=5)
    # disp1 = ConfusionMatrixDisplay(cm1)
    # disp1.plot(cmap=plt.cm.Blues)
    # plt.savefig(path_images + "/output2.png")
    # file_results.write("Result of SVM with operands opcode\n")
    # file_results.write(str(report1))

    # ensembling model with no operand opcode
    est_XGB = xgboost.XGBClassifier()
    est_BA = ExtraTreesClassifier()
    est_Ensemble = VotingClassifier(
        estimators=[("est_XGB", est_XGB), ("est_BA", est_BA)],
        voting="soft",
        weights=[1, 1],
    )
    # est_Ensemble.fit(x_train, y_train)
    # pred_ensemble = est_Ensemble.predict(x_test)
    # cm_ensemble = confusion_matrix(y_test, y_pred=pred_ensemble)
    # report2 = classification_report(y_test, pred_ensemble, digits=5)
    # disp2 = ConfusionMatrixDisplay(cm_ensemble)
    # disp2.plot(cmap=plt.cm.Blues)
    # plt.savefig(path_images + "/output3.png")
    # file_results.write("Result of ensemble with no operands opcode\n")
    # file_results.write(str(report2))

    # Add value feature
    scaled = (
        df["value"].apply(lambda x: float(x)) / 1000000000000000000
    ).values.reshape(-1, 1)
    a = coo_matrix(Data_noOp)
    b = coo_matrix(scaled)
    scaled_val = hstack([a, b]).toarray()
    scaled = csr_matrix(scaled_val)

    # Ensemble model
    x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(
        scaled, y, test_size=0.2, random_state=0
    )

    batch_size = 500
    for i in range(0, x_train_val.shape[0], batch_size):
        X_batch = x_train_val[i : i + batch_size, :]
        y_batch = y_train_val[i : i + batch_size]
        est_Ensemble.fit(X_batch, y_batch)

    pred_val = est_Ensemble.predict(x_test_val)
    cm_val = confusion_matrix(y_test_val, pred_val)
    reportVal = classification_report(y_test_val, pred_val)
    labels = ["Delegatecall", "DoS", "FDV", "FoT", "Normal", "OaU"]
    dispVal = ConfusionMatrixDisplay(cm_val, display_labels=labels)
    dispVal.plot(cmap=plt.cm.Blues)
    plt.savefig(path_images + "/output.png")
    file_results.write("Result of Ensemble with no operands opcode and value\n")
    file_results.write(str(reportVal))

    # SVM
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    svm = LinearSVC(random_state=42)
    true_labels = []
    predicted_labels = []
    for train_indices, test_indices in kfold.split(scaled):
        X_train_sparse = scaled[train_indices]
        X_test_sparse = scaled[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

    svm.fit(X_train_sparse, y_train)
    y_pred = svm.predict(X_test_sparse)
    true_labels.extend(y_test)
    predicted_labels.extend(y_pred)
    reportVal2 = classification_report(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    cmd = ConfusionMatrixDisplay(cm, display_labels=svm.classes_)
    cmd.plot(cmap=plt.cm.Blues)
    plt.savefig(path_images + "/output1.png")
    file_results.write("Result of SVM with no operands opcode and value\n")
    file_results.write(str(reportVal2))
    file_results.close()


if __name__ == "__main__":
    path_images = os.path.dirname(os.path.realpath(__file__))
    tx_list = []
    label = []
    path1 = "./Delegatecall"
    path2 = "./DoS"
    path3 = "./FDV"
    path4 = "./FoT"
    path5 = "./Normal"
    path6 = "./OaU"

    tx_path1 = getFiles(path1)
    tx_path2 = getFiles(path2)
    tx_path3 = getFiles(path3)
    tx_path4 = getFiles(path4)
    tx_path5 = getFiles(path5)
    tx_path6 = getFiles(path6)

    tx_paths = tx_path1 + tx_path2 + tx_path3 + tx_path4 + tx_path5 + tx_path6

    output = "./output.txt"

    main(tx_paths, tx_list, label, output, path_images)
