# -*- coding: utf-8 -*-

import os
from collections import Counter

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

from config import Config


def get_wavs_lables():
    conf = Config()
    wav_files, text_labels = do_get_wavs_lables(conf.get("FILE_DATA").wav_path,
                                                conf.get("FILE_DATA").label_file)
    print(wav_files[0], text_labels[0])
    # wav/train/A11/A11_0.WAV -> 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然
    print("wav:", len(wav_files), "label", len(text_labels))

    return wav_files, text_labels


def do_get_wavs_lables(wav_path, label_file):
    """
    读取wav文件对应的label
    :param wav_path:
    :param label_file:
    :return:
    """
    # 获得训练用的wav文件路径列表
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:  # 剔除掉一些小文件
                    continue
                wav_files.append(filename_path)

    labels_dict = {}
    with open(label_file, 'rb') as f:
        for label in f:
            label = label.strip(b'\n')
            label_id = label.split(b' ', 1)[0]
            label_text = label.split(b' ', 1)[1]
            labels_dict[label_id.decode('ascii')] = label_text.decode('utf-8')

    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        # print(os.path.basename(wav_file))
        wav_id = os.path.basename(wav_file).split('.')[0]

        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    return new_wav_files, labels


def create_dict(text_labels):
    """
    构建字典
    :param text_labels:
    :return:
    """
    all_words = []
    for label in text_labels:
        # print(label)
        all_words += [word for word in label]
    counter = Counter(all_words)
    words = sorted(counter)
    words_size = len(words)
    word_num_map = dict(zip(words, range(words_size)))
    print('字表大小:', words_size)

    return words_size, words, word_num_map


def next_batch(start_idx=0,
               batch_size=1,
               n_input=None,
               n_context=None,
               labels=None,
               wav_files=None,
               word_num_map=None):
    """
    按批次获取样本
    :param start_idx:
    :param batch_size:
    :param n_input:
    :param n_context:
    :param labels:
    :param wav_files:
    :param word_num_map:
    :return:
    """
    filesize = len(labels)
    end_idx = min(filesize, start_idx + batch_size)
    idx_list = range(start_idx, end_idx)
    txt_labels = [labels[i] for i in idx_list]
    wav_files = [wav_files[i] for i in idx_list]
    audio_features, audio_features_len, text_vector, text_vector_len = get_audio_mfcc_features(None,
                                                                                               wav_files,
                                                                                               n_input,
                                                                                               n_context,
                                                                                               word_num_map,
                                                                                               txt_labels)

    start_idx += batch_size
    # 验证 start_idx
    if start_idx >= filesize:
        start_idx = -1

    # 如果多个文件将长度统一，支持按最大截断或补0
    audio_features, audio_features_len = pad_sequences(audio_features)
    sparse_labels = sparse_tuple_from(text_vector)

    return start_idx, audio_features, audio_features_len, sparse_labels, wav_files


def get_audio_mfcc_features(txt_files, wav_files, n_input, n_context, word_num_map, txt_labels=None):
    """
    提取音频数据的MFCC特征
    :param txt_files:
    :param wav_files:
    :param n_input:
    :param n_context:
    :param word_num_map:
    :param txt_labels:
    :return:
    """
    audio_features = []
    audio_features_len = []
    text_vector = []
    text_vector_len = []
    if txt_files != None:
        txt_labels = txt_files

    for txt_obj, wav_file in zip(txt_labels, wav_files):
        # 载入音频数据并转化为特征值
        audio_data = audiofile_to_input_vector(wav_file, n_input, n_context)
        audio_data = audio_data.astype('float32')

        audio_features.append(audio_data)
        audio_features_len.append(np.int32(len(audio_data)))

        # 载入音频对应的文本
        target = []
        if txt_files != None:  # txt_obj是文件
            target = trans_text_ch_to_vector(txt_obj, word_num_map)
        else:
            target = trans_text_ch_to_vector(None, word_num_map, txt_obj)  # txt_obj是labels
        # target = text_to_char_array(target)
        text_vector.append(target)
        text_vector_len.append(len(target))

    audio_features = np.asarray(audio_features)
    audio_features_len = np.asarray(audio_features_len)
    text_vector = np.asarray(text_vector)
    text_vector_len = np.asarray(text_vector_len)
    return audio_features, audio_features_len, text_vector, text_vector_len


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    密集矩阵转稀疏矩阵
    :param sequences:
    :param dtype:
    :return:
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    # temp = indices.max(0)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    return indices, values, shape


def trans_text_ch_to_vector(txt_file, word_num_map, txt_label=None):
    """
    中文字符到向量
    :param txt_file:
    :param word_num_map:
    :param txt_label:
    :return:
    """
    words_size = len(word_num_map)

    to_num = lambda word: word_num_map.get(word, words_size)

    if txt_file != None:
        txt_label = get_ch_lable(txt_file)

    # print(txt_label)
    labels_vector = list(map(to_num, txt_label))
    # print(labels_vector)
    return labels_vector


def get_ch_lable(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            # labels =label.decode('utf-8')
            labels = labels + label.decode('gb2312')
            # labels.append(label.decode('gb2312'))

    return labels


def trans_tuple_to_texts_ch(tuple, words):
    """
    向量转换成文字
    :param tuple:
    :param words:
    :return:
    """
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    #print('word len is:' , len(words))
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == 0 else words[c]  # chr(c + FIRST_INDEX)
        results[index] = results[index] + c

    return results


def trans_array_to_text_ch(value, words):
    results = ''
    #print('trans_array_to_text_ch len:', len(value))
    for i in range(len(value)):
        results += words[value[i]]  # chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')


def audiofile_to_input_vector(audio_filename, n_input, n_context):
    """
    将音频装换成MFCC
    :param audio_filename:
    :param n_input:
    :param n_context:
    :return:
    """
    # 加载wav文件
    fs, audio = wav.read(audio_filename)

    # 获取mfcc数值
    orig_inputs = mfcc(audio, samplerate=fs, numcep=n_input)
    # print(np.shape(orig_inputs))  #(277, 26)
    orig_inputs = orig_inputs[::2]  # (139, 26) 每隔一行进行一次取样

    # train_inputs = np.array([], np.float32)
    # print(orig_inputs.shape[0])
    train_inputs = np.zeros((orig_inputs.shape[0], n_input + 2 * n_input * n_context))
    # print(np.shape(train_inputs))#)(139, 494)
    # empty_mfcc = np.array([])
    empty_mfcc = np.zeros((n_input))

    # 准备输入数据，数据由三部分安顺序拼接而成，分为当前样本的前9个序列样本，当前样本序列，后9个序列样本
    time_slices = range(train_inputs.shape[0])  # 139个切片
    context_past_min = time_slices[0] + n_context
    context_future_max = time_slices[-1] - n_context  # [9,1,2...,137,129]
    for time_slice in time_slices:
        # 前9个补0，mfcc features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - n_context):time_slice]

        # 后9个补0，mfcc features
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + n_context + 1]

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, n_context * n_input)
        now = orig_inputs[time_slice]
        future = np.reshape(future, n_context * n_input)
        # 234, 26, 234
        # train_data = np.concatenate((past, now, future));
        train_inputs[time_slice] = np.concatenate((past, now, future))

    # 将数据使用正太分布标准化，减去均值然后再除以方差
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    """
    音频数据对齐
    post表示后补0  pre表示前补0
    :param sequences:
    :param maxlen:
    :param dtype:
    :param padding:
    :param truncating:
    :param value:
    :return:
    """
    sequences_each_len = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(sequences_each_len)

    # 从第一个非空的序列中的样本形状
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            # test
            # temp = np.asarray(s)
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, sequences_each_len


if __name__ == "__main__":
    conf = Config()

    get_wavs_lables(conf.get("FILE_DATA").wav_path, conf.get("FILE_DATA").label_file)
    print()
