import numpy as np
import os
import utils
import copy
import random
import time
import json
# from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import argparse
from joblib import dump, load
from constants import *
import re
import pandas as pd
import sys
import matplotlib.pyplot as plt


def filter_raw_data(sample_left, sample_right):
    filtered_row = []
    # We need: for RMS: z - axis acc for std dev: z - axis acc for FI: x / y / z - axis acc
    # filtered_row values = time(left), left_leg_acc_X - Y - Z/right_leg_acc_X - Y - Z, FOG mark(left)
    indices_left = {
        "time": 0,
        "left_accx": 4,
        "left_accy": 5,
        "left_accz": 6,
        "FoG": 10
    }
    indices_right = {
        "right_accx": 4,
        "right_accy": 5,
        "right_accz": 6,
    }
    insert_index = 4
    for key, value in indices_left.items():
        filtered_row.append(sample_left[value])
    for key, value in indices_right.items():
        filtered_row.insert(insert_index, sample_right[value])
        insert_index += 1
    return filtered_row


# input is win_size rows of 8 data values
def window_sum(window):
    sum = 0
    for i in range(len(window)):
        fog_value = abs(int(float(window[i][FOG_INDEX])))
        sum += fog_value
    return sum


def form_windows(patient_filtered_trial, win_size, store_fog_list, store_non_fog_list):
    i = win_size
    fog_count = 0
    non_fog_count = 0

    while i <= len(patient_filtered_trial):
        current_window = patient_filtered_trial[i-win_size: i]
        window_fog_sum = window_sum(current_window)
        # FoG window
        if window_fog_sum == win_size:
            fog_count += 1
            store_fog_list.append(current_window)
            # i += win_size
        # Non-FoG window
        elif window_fog_sum == 0:
            non_fog_count += 1
            store_non_fog_list.append(current_window)
            # i += win_size
        # else:
        #     i += 1
        i += 1


def form_windows_latest_label(patient_filtered_trial, win_size, store_fog_list, store_non_fog_list):
    i = win_size
    fog_count = 0
    non_fog_count = 0

    while i <= len(patient_filtered_trial):
        current_window = patient_filtered_trial[i - win_size: i]
        window_label = int(float(current_window[win_size-1][FOG_INDEX]))
        # FoG window
        if window_label == 1:
            fog_count += 1
            store_fog_list.append(current_window)
            # i += win_size
        # Non-FoG window
        elif window_label == 0:
            non_fog_count += 1
            store_non_fog_list.append(current_window)
            # i += win_size
        # else:
        #     i += 1
        i += 1


def extract_windows_leave_one_out(main_config, patient_left_out):
    data_stats = {}
    move_file_lookup = {}
    s1exp = re.compile(r's1.csv')
    s2exp = re.compile(r's2.csv')
    s3exp = re.compile(r's3.csv')
    xlsexp = re.compile(r'.xls')
    window_size = main_config["win_size"]
    # list of windows
    store_fog = []
    store_non_fog = []
    patient_index = 0
    # patient_left_out = random.randrange(1, 64, 1)
    patient_left_out_trials = []

    for root, dirs, files in os.walk(DATASET_PATH, topdown=False):
        for patient_folder in dirs:
            patient_folder_path = os.path.join(root, patient_folder)
            patient_filtered = []
            left_leg = []
            right_leg = []
            can_filter = False
            patient_index += 1
            for file in os.listdir(patient_folder_path):
                # print(file)
                file_path = patient_folder_path + '\\' + file
                # print(file_path)
                pre, ext = os.path.splitext(file_path)

                if "s1" in pre:
                    # print(file)
                    continue

                # filter left (s2) and right (s3) leg data
                patient_raw_data = utils.read_csv(file_path, load_header=False, delimiter=",")
                # print(patient_raw_data)

                if s2exp.search(file):
                    # print(file)
                    left_leg = patient_raw_data

                elif s3exp.search(file):
                    # print(file)
                    right_leg = patient_raw_data
                    can_filter = True

                if can_filter:
                    num_samples = min(len(left_leg), len(right_leg))
                    # print("num samples for trial {}".format(num_samples))
                    for i in range(num_samples):
                        patient_trial_data_row = filter_raw_data(left_leg[i], right_leg[i])
                        patient_filtered.append(patient_trial_data_row)
                    can_filter = False

                    if patient_left_out == patient_index:
                        patient_left_out_trials.append(patient_filtered)
                    else:
                        # consolidate window for trial
                        form_windows_latest_label(patient_filtered, window_size, store_fog, store_non_fog)
                    patient_filtered = []

    # testing
    # 1673 in matlab
    print(len(store_fog))
    # 3124 in matlab
    print(len(store_non_fog))
    print("patient left out: {}".format(patient_left_out))
    print("patient trials: {}".format(len(patient_left_out_trials)))

    return store_fog, store_non_fog, patient_left_out_trials


def extract_windows(main_config):
    data_stats = {}
    move_file_lookup = {}
    s1exp = re.compile(r's1.csv')
    s2exp = re.compile(r's2.csv')
    s3exp = re.compile(r's3.csv')
    xlsexp = re.compile(r'.xls')
    window_size = main_config["win_size"]
    # list of windows
    store_fog = []
    store_non_fog = []

    for root, dirs, files in os.walk(DATASET_PATH, topdown=False):
        for patient_folder in dirs:
            patient_folder_path = os.path.join(root, patient_folder)
            patient_filtered = []
            left_leg = []
            right_leg = []
            can_filter = False
            for file in os.listdir(patient_folder_path):
                # print(file)
                file_path = patient_folder_path + '\\' + file
                # print(file_path)
                pre, ext = os.path.splitext(file_path)

                if "s1" in pre:
                    # print(file)
                    continue

                # filter left (s2) and right (s3) leg data
                patient_raw_data = utils.read_csv(file_path, load_header=False, delimiter=",")
                # print(patient_raw_data)

                if s2exp.search(file):
                    # print(file)
                    left_leg = patient_raw_data

                elif s3exp.search(file):
                    # print(file)
                    right_leg = patient_raw_data
                    can_filter = True

                if can_filter:
                    num_samples = min(len(left_leg), len(right_leg))
                    # print("num samples for trial {}".format(num_samples))
                    for i in range(num_samples):
                        patient_trial_data_row = filter_raw_data(left_leg[i], right_leg[i])
                        patient_filtered.append(patient_trial_data_row)
                    can_filter = False
                    # consolidate window for trial
                    # form_windows(patient_filtered, window_size, store_fog, store_non_fog)
                    form_windows_latest_label(patient_filtered, window_size, store_fog, store_non_fog)
                    patient_filtered = []

    # testing
    # 1673 in matlab
    print(len(store_fog))
    # 3124 in matlab
    print(len(store_non_fog))

    return store_fog, store_non_fog


def feature_extract(train_windows, test_windows):
    features_train_windows = []
    features_train_windows = np.empty(shape=(len(train_windows), 3))
    labels_train = np.empty(shape=(len(train_windows),))
    features_test_windows = []
    features_test_windows = np.empty(shape=(len(test_windows), 3))
    labels_test = np.empty(shape=(len(test_windows),))

    # calculate RMS, Standard Deviation, Freeze Index for every window
    for i in range(len(train_windows)):
        window = train_windows[i]
        features_window = []

        left_accz_rms = utils.extract_rms(window, LEFT_ACCZ_INDEX)
        right_accz_rms = utils.extract_rms(window, RIGHT_ACCZ_INDEX)
        avg_accz_rms = (left_accz_rms + right_accz_rms) / 2.0
        features_window.append(avg_accz_rms)

        left_accz_std = utils.extract_std(window, LEFT_ACCZ_INDEX)
        right_accz_std = utils.extract_std(window, RIGHT_ACCZ_INDEX)
        avg_accz_std = (left_accz_std + right_accz_std) / 2.0
        features_window.append(avg_accz_std)

        features_window.append(utils.extract_fi(window, LEFT_ACCX_INDEX, RIGHT_ACCX_INDEX, LB_LOW, LB_HIGH, FB_LOW, FB_HIGH))

        # features_train_windows.append(features_window)
        features_train_windows[i] = np.array(features_window)

        labels_train[i] = int(float(window[0][FOG_INDEX]))

    for j in range(len(test_windows)):
        window = test_windows[j]
        features_window = []

        left_accz_rms = utils.extract_rms(window, LEFT_ACCZ_INDEX)
        right_accz_rms = utils.extract_rms(window, RIGHT_ACCZ_INDEX)
        avg_accz_rms = (left_accz_rms + right_accz_rms) / 2.0
        features_window.append(avg_accz_rms)

        left_accz_std = utils.extract_std(window, LEFT_ACCZ_INDEX)
        right_accz_std = utils.extract_std(window, RIGHT_ACCZ_INDEX)
        avg_accz_std = (left_accz_std + right_accz_std) / 2.0
        features_window.append(avg_accz_std)

        features_window.append(
            utils.extract_fi(window, LEFT_ACCX_INDEX, RIGHT_ACCX_INDEX, LB_LOW, LB_HIGH, FB_LOW, FB_HIGH))

        # features_test_windows.append(features_window)
        features_test_windows[j] = np.array(features_window)

        labels_test[j] = int(float(window[0][FOG_INDEX]))

    # print("Length of each sample features: {0:5d}".format(len(features_train_windows[0])))

    return features_train_windows, labels_train, features_test_windows, labels_test


def form_test_samples(test_windows, window_size):
    num_samples = len(test_windows) * window_size
    test_samples = np.empty(shape=(num_samples, 8))
    count = 0
    for i in range(len(test_windows)):
        window = test_windows[i]
        for j in range(window_size):
            test_samples[count] = np.array(window[j])
            count += 1

    return test_samples


def simulate_sensor(clf, window_size, test_windows, freq):
    left_ind = 0
    right_ind = left_ind + window_size

    test_samples = form_test_samples(test_windows, window_size)

    predicted_labels = []
    avg_labels = []
    # form window
    while right_ind <= len(test_samples):
        window = test_samples[left_ind:right_ind]
        features_window = []
        sample = np.empty(shape=(1, 3))

        # append actual average label of window
        avg_labels.append(utils.extract_avg_label(window, FOG_INDEX))

        # feature extract
        left_accz_rms = utils.extract_rms(window, LEFT_ACCZ_INDEX)
        right_accz_rms = utils.extract_rms(window, RIGHT_ACCZ_INDEX)
        avg_accz_rms = (left_accz_rms + right_accz_rms) / 2.0
        features_window.append(avg_accz_rms)

        left_accz_std = utils.extract_std(window, LEFT_ACCZ_INDEX)
        right_accz_std = utils.extract_std(window, RIGHT_ACCZ_INDEX)
        avg_accz_std = (left_accz_std + right_accz_std) / 2.0
        features_window.append(avg_accz_std)

        features_window.append(
            utils.extract_fi(window, LEFT_ACCX_INDEX, RIGHT_ACCX_INDEX, LB_LOW, LB_HIGH, FB_LOW, FB_HIGH))

        # predict with 1 sample
        sample[0] = np.array(features_window)
        predicted_label = clf.predict(sample)
        predicted_labels.append(predicted_label[0])

        left_ind += window_size
        right_ind += window_size

        if left_ind > 1000:
            break

    print(avg_labels)
    print(predicted_labels)
    # plt.plot(avg_labels, label='average_label of window')
    # plt.plot(predicted_labels, label='predicted labels')
    # plt.title('Comparison between predicted and avg label')
    # plt.legend(loc='best')
    # plt.show()
    sys.exit(0)


def simulate_leave_one_out(clf, window_size, test_trials, predicted_labels, avg_labels):
    for i in range(len(test_trials)):
        trial = test_trials[i]

        left_ind = 0
        right_ind = left_ind + window_size

        # form window
        while right_ind <= len(trial):
            window = trial[left_ind:right_ind]
            features_window = []
            sample = np.empty(shape=(1, 3))

            # append actual average label of window
            # avg_labels.append(utils.extract_avg_label(window, FOG_INDEX))
            avg_labels.append(int(float(window[len(window)-1][FOG_INDEX])))

            # feature extract
            left_accz_rms = utils.extract_rms(window, LEFT_ACCZ_INDEX)
            right_accz_rms = utils.extract_rms(window, RIGHT_ACCZ_INDEX)
            avg_accz_rms = (left_accz_rms + right_accz_rms) / 2.0
            features_window.append(avg_accz_rms)

            left_accz_std = utils.extract_std(window, LEFT_ACCZ_INDEX)
            right_accz_std = utils.extract_std(window, RIGHT_ACCZ_INDEX)
            avg_accz_std = (left_accz_std + right_accz_std) / 2.0
            features_window.append(avg_accz_std)

            features_window.append(
                utils.extract_fi(window, LEFT_ACCX_INDEX, RIGHT_ACCX_INDEX, LB_LOW, LB_HIGH, FB_LOW, FB_HIGH))

            # predict with 1 sample
            sample[0] = np.array(features_window)
            predicted_label = clf.predict(sample)
            predicted_labels.append(predicted_label[0])

            left_ind += 1
            right_ind += 1


    # print(avg_labels)
    # print(predicted_labels)
    # plt.plot(avg_labels, label='average_label of window')
    # plt.plot(predicted_labels, label='predicted labels')
    # plt.title('Comparison between predicted and avg label')
    # plt.legend(loc='best')
    # plt.show()


def main(config):
    data_dir = "data"
    data_limits = [0.2, 0.4, 0.6, 0.8, 1.0]
    fog_windows, non_fog_windows = extract_windows(config)
    fog_windows_train, fog_windows_test = train_test_split(fog_windows, test_size=config["test_size"])
    non_fog_percentage_train = len(fog_windows_train) / len(non_fog_windows)
    non_fog_windows_train, non_fog_windows_remain = train_test_split(non_fog_windows, train_size=non_fog_percentage_train)
    # non_fog_windows_train, non_fog_windows_remain = train_test_split(non_fog_windows, train_size=0.38)
    # non_fog_windows_train, non_fog_windows_test = train_test_split(non_fog_windows, test_size=config["test_size"])
    non_fog_percentage_test = len(fog_windows_test) / len(non_fog_windows_remain)
    non_fog_windows_test, non_fog_windows_excess = train_test_split(non_fog_windows_remain, train_size=non_fog_percentage_test)
    # non_fog_windows_test, non_fog_windows_excess = train_test_split(non_fog_windows_remain, train_size=0.26)
    train_windows = []
    test_windows = []
    train_windows.extend(fog_windows_train)
    train_windows.extend(non_fog_windows_train)
    test_windows.extend(fog_windows_test)
    test_windows.extend(non_fog_windows_test)
    print(len(fog_windows_train))
    print(len(non_fog_windows_train))
    print(len(fog_windows_test))
    print(len(non_fog_windows_test))
    features_train, labels_train, features_test, labels_test = feature_extract(train_windows, test_windows)

    if config["model_type"] == "lda":
        if config["mode"] == "train_and_test":
            clf = LinearDiscriminantAnalysis()
            # print(labels_train.shape)
            # print(labels_train[0])
            # print(features_train.shape)
            # print(len(features_train[0]))
            # print(len(features_train))
            # print(len(labels_train))
            clf.fit(features_train, labels_train)
        elif config["mode"] == "simulate":
            clf = load('lda_sliding.joblib')
            simulate_sensor(clf, config["win_size"], test_windows, 50)

        # score = clf.score(features_test, labels_test)
        predicted_labels = clf.predict(features_test)
        target_names = ['non-FOG', 'FOG']
        print(classification_report(labels_test, predicted_labels, target_names=target_names))
        # score = utils.check_accuracy(predicted_labels, labels_test)
        # print("Score: {0:.3f}".format(score))
        dump(clf, "lda.joblib")


def leave_one_out_main(config):
    predicted_labels = []
    avg_labels = []
    for i in range(63):
        fog_windows_train, non_fog_windows,  patient_left_out_trials = extract_windows_leave_one_out(config, i+1)
        # print(len(fog_windows_train))
        # print(len(non_fog_windows))
        non_fog_windows_train_percentage = len(fog_windows_train) / len(non_fog_windows)
        non_fog_windows_train, non_fog_windows_remain = train_test_split(non_fog_windows, train_size=non_fog_windows_train_percentage)
        train_windows = []
        train_windows.extend(fog_windows_train)
        train_windows.extend(non_fog_windows_train)
        features_train, labels_train, features_test, labels_test = feature_extract(train_windows, [])
        clf = LinearDiscriminantAnalysis()
        clf.fit(features_train, labels_train)
        simulate_leave_one_out(clf, config["win_size"], patient_left_out_trials, predicted_labels, avg_labels)
        print("finished patient {}".format(i+1))

    target_names = ['non-FOG', 'FOG']
    print(classification_report(avg_labels, predicted_labels, target_names=target_names))


def parse_args():
    parser = argparse.ArgumentParser(description='Community FAQ pre-processing')
    parser.add_argument('-s', '--simulate', action='store_true', help='Only simulate')
    return parser.parse_args()


if __name__ == "__main__":
    p_args = parse_args()
    config = {
        "win_size": 100,
        "min_confidence": 0.5,
        "mode": "train_and_test",
        "model_type": "lda",
        "test_size": 0.3
    }
    if p_args.simulate:
        # test_files = utils.load_text_as_list("test_files.txt")
        # trained_rf = joblib.load("rf.joblib")
        # trained_mlp = joblib.load
        print("in simulation mode...")
        config["mode"] = "simulate"

    main(config)