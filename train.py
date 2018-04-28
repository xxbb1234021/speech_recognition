# -*- coding: utf-8 -*-

import os

import utils
from config import Config
from model import BiRNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conf = Config()

wav_files, text_labels = utils.get_wavs_lables()

words_size, words, word_num_map = utils.create_dict(text_labels)


bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
bi_rnn.build_train()
