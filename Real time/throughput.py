import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
from gensim.models import Word2Vec
import os
import json
import csv

web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
vocab = joblib.load('vocab.pkl')
model = load_model('final_model.h5')
modelWord = Word2Vec.load('w2v.model')

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

def word2vec(op_str):
    tokens = op_str.split()
    # train Word2Vec model on the list of opcode tokens
    model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)
    # get vector representation of each token
    vectors = []
    for token in tokens:
        try:
            token_vec = model.wv[token]
            vectors.append(token_vec)
        except KeyError:
            continue
    # compute the mean vector of all the token vectors
    if len(vectors) > 0:
        op_vec = np.mean(vectors, axis=0)
    else:
        op_vec = np.zeros(50) # return a zero vector if no tokens are found
    return op_vec

def combineFeatures(data, value):
    value = np.array(value)
    value_coo = coo_matrix(value)
    Data_coo = coo_matrix(data)
    matrix = hstack([Data_coo,value_coo]).toarray()
    Fea_matrix = csr_matrix(matrix)
    return matrix


processed_tx = 0
printed_txs = set()
num_tx = 50 
tx_dir = "./Normal_Json"
tx_processed = 0
count = 0
total_time = 0
csv_file = 'throughput.csv'

with open(csv_file, mode='w', newline='') as log_file:
    log_writer = csv.writer(log_file)

    while True:
        count +=1
        #pending_block = web3.eth.get_block("pending")
        #pending_txs = pending_block["transactions"]
        #new_txs_raw = [tx for tx in pending_txs if tx not in printed_txs]

        tx_files = [f for f in os.listdir(tx_dir) if f.endswith('.json')]

        if len(tx_files) < num_tx:
            break

        new_tx_files = tx_files[tx_processed:tx_processed+num_tx]
        tx_processed += num_tx
        if tx_processed > len(tx_files):
            tx_processed = len(tx_files)
        
        #print(new_tx_files)
    

        inputs = []
        values = []

        for tx_file in new_tx_files:
            with open(os.path.join(tx_dir, tx_file)) as f:
                tx_data = json.load(f)
                inputs.append(tx_data['input'])
                values.append(tx_data['value'])

        #print(inputs)
        
        
        df = pd.DataFrame(inputs,columns=['Input'])
        df['Value'] = values

        start_time = time.time()
        print('Start predict at:',start_time)
        pre_text = df['Input'].apply(preprocess_text)
        text = pre_text.apply(BytecodeToOpcode)
        opcode = text.apply(clean_text)
        pp_opcode = opcode.apply(preprocess_opcode)
        
        cv = TfidfVectorizer(tokenizer= dummy, preprocessor= dummy, analyzer='word',token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=30000,vocabulary=vocab)
        
        try:
            input_vec = cv.fit_transform(pp_opcode)
        except ValueError:
            continue    


        col_values = (df['Value'].apply(lambda x: float(x))/1000000000000000000).values.reshape(-1,1)
        Vec = combineFeatures(input_vec,col_values)
    


        class_mapping = {0: "Delegatecall", 1: "DoS", 2: "FDV", 3: "FoT", 4: "Normal", 5: "OaU"}

        prediction = model.predict(Vec,verbose = 0)

        predicted_class = np.argmax(prediction, axis=-1)
        predicted_class_string = [class_mapping[x] for x in predicted_class]
        

        print('Predicted class:', predicted_class_string)

        #txs_processed = len(new_txs_raw)
        #processed_tx += txs_processed
        end_time = time.time()
        process_time = end_time - start_time
        total_time += process_time
        print(f"Processed {num_tx*count} tx in {total_time:.2f} seconds")
        log_writer.writerow([num_tx*count, total_time])
        inputs.clear()
        values.clear()
        time.sleep(1)
log_file.close()
