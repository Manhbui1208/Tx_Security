import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
import sys
import keras
from keras.models import load_model, Sequential
from keras.layers import Dense
from pyevmasm import instruction_tables, disassemble_hex, disassemble_all, assemble_hex
import binascii
import gensim
import re
import numpy as np
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, hstack, coo_matrix
import joblib
import time
#nltk.download('wordnet')
from web3 import Web3
import pandas as pd
import threading
import random 
from datetime import datetime

web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
vocab = joblib.load('vocab.pkl')
model = load_model('final_model.h5')

def opcode_to_vector(op_str, model):
    tokens = op_str.split()
    vectors = []
    for token in tokens:
        try:
            token_vec = model.wv[token]
            vectors.append(token_vec)
        except KeyError:
            continue
    if len(vectors) > 0:
        op_vec = np.mean(vectors, axis=0)
    else:
        op_vec = np.zeros(100)
    return op_vec

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


def tfidf(data):
    cv = TfidfVectorizer(tokenizer= dummy, preprocessor= dummy, analyzer='word',token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000,vocabulary=vocab)
    Data = cv.fit_transform(data)
    return Data

def combineFeatures(data, value):
    value = np.array(value)
    value_coo = coo_matrix(value)
    Data_coo = coo_matrix(data)
    matrix = hstack([Data_coo,value_coo]).toarray()
    Fea_matrix = csr_matrix(matrix)
    return matrix

timestamps = []
normal_counts = []
fot_counts = []
dos_counts = []
dec_counts = []
oau_counts = []
fdv_counts = []
printed_txs = set()


class RealTimePlotWidget(QMainWindow):
    def __init__(self):
        super(RealTimePlotWidget, self).__init__()

        loadUi('test.ui', self)

        self.widget = self.findChild(QWidget, 'widget')

        self.data_thread = threading.Thread(target=self.process_realtime_data)
        self.data_thread.daemon = True
        self.data_thread.start()
  
        self.fig = Figure()

        # Tạo đối tượng FigureCanvas để hiển thị đồ thị
        self.canvas = FigureCanvas(self.fig)

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.update_plot(timestamps, normal_counts, fot_counts, dos_counts,dec_counts,oau_counts,fdv_counts))
        self.timer.start(1000)
        
        layout = QVBoxLayout(self.widget)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)



    def update_plot(self,timestamp,normal_count,fot_count,dos_count,dec_count,oau_count,fdv_count):
        if len(timestamp) > 1000:
            timestamp.pop(0)
            normal_count.pop(0)
            fot_count.pop(0)
            dos_count.pop(0)
            dec_count.pop(0)
            oau_count.pop(0)
            fdv_count.pop(0)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(timestamp, normal_count, label='Normal', marker='o')
        ax.plot(timestamp, fot_count, label='FoT', marker='s')
        ax.plot(timestamp, dos_count, label='DoS', marker='d')
        ax.plot(timestamp, dec_count, label='DeC', marker='^')
        ax.plot(timestamp, oau_count, label='OaU', marker='*')
        ax.plot(timestamp, fdv_count, label='FDV', marker='p')
        ax.set_xlabel('Time')
        ax.set_ylabel('Num of Tx')
        #ax.set_title('Real-time Detection Flow')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.grid()  # Thêm major grid
        ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=6, frameon=False) 
        self.canvas.draw()

    def process_realtime_data(self):
        processed_tx = 0
        elapsed_time = 0
        while True:
            pending_block = web3.eth.get_block("pending")
            pending_txs = pending_block["transactions"]
            new_txs_raw = [tx for tx in pending_txs if tx not in printed_txs]

            inputs = []
            values = []

            for tx_hash in new_txs_raw:
                input_data = web3.eth.get_transaction(tx_hash)['input']
                value_data = web3.eth.get_transaction(tx_hash)['value']
                inputs.append(input_data)
                values.append(value_data)

            df = pd.DataFrame(inputs,columns=['Input'])
            df['Value'] = values

            start_time = time.time()
            print('Start predict at:',start_time)
            pre_text = df['Input'].apply(preprocess_text)
            text = pre_text.apply(BytecodeToOpcode)
            opcode = text.apply(clean_text)
            pp_opcode = opcode.apply(preprocess_opcode)
            #pp_opcode1 = pp_opcode.apply(lambda x: tuple(x))
            #print(pp_opcode1)
            cv = TfidfVectorizer(tokenizer= dummy, preprocessor= dummy, analyzer='word',token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000,vocabulary=vocab)
            
            try:
                input_vec = cv.fit_transform(pp_opcode)
            except ValueError:
                continue    

            col_values = (df['Value'].apply(lambda x: float(x))/1000000000000000000).values.reshape(-1,1)
            Vec = combineFeatures(input_vec,col_values)

            class_mapping = {0: "Delegatecall", 1: "DoS", 2: "FDV", 3: "FoT", 4: "Normal", 5: "OaU"}
            class_count = {class_name: 0 for class_name in class_mapping.values()}
            prediction = model.predict(Vec,verbose = 0)
            predicted_class = np.argmax(prediction, axis=-1)
            predicted_class_string = [class_mapping[x] for x in predicted_class]
            print('Predicted class:', predicted_class_string)
            for element in predicted_class_string:
                class_count[element] += 1
            
            print('Num of FoT:',class_count['FoT'])
            print('Num of Nor:',class_count['Normal'])
            print('Num of DoS:',class_count['DoS'])
            print('Num of DeC:',class_count['Delegatecall'])
            print('Num of FDV:',class_count['FDV'])
            print('Num of OaU:',class_count['OaU'])
            fot_counts.append(class_count['FoT'])
            normal_counts.append(class_count['Normal'])
            dos_counts.append(class_count['DoS'])
            dec_counts.append(class_count['Delegatecall'])
            fdv_counts.append(class_count['FDV'])
            oau_counts.append(class_count['OaU'])
            timestamps.append(int(start_time))

            txs_processed = len(new_txs_raw)
            processed_tx += txs_processed

            end_time = time.time()
            
            elapsed_time += (end_time - start_time)
            print(elapsed_time)
        
            if elapsed_time < 1:
                continue
            else:
                tx_per_second = processed_tx / elapsed_time
                print("Processed {} tx in {:.2f} seconds ({:.2f} tx/s)".format(processed_tx, elapsed_time, tx_per_second))
                start_time = end_time
                processed_tx = 0
                elapsed_time = 0

            printed_txs.update(new_txs_raw)
            inputs.clear()
            values.clear()
            time.sleep(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RealTimePlotWidget()
    window.show()
    sys.exit(app.exec_())
