TRAIN_DEV_FILES = ["subject3_ideal.log", "subject4_ideal.log"]
TEST_SIZE = 0.1
DATASET_PATH = "C:\\Users\\Wen Hao\\PycharmProjects\\test\\4_FoG_dataset_63p"
FOG_VALUE = 1
NON_FOG_VALUE = 0

# Indices in window
LEFT_ACCX_INDEX = 1
LEFT_ACCY_INDEX = 2
LEFT_ACCZ_INDEX = 3
RIGHT_ACCX_INDEX = 4
RIGHT_ACCY_INDEX = 5
RIGHT_ACCZ_INDEX = 6
FOG_INDEX = 7

INCLUDED_FEATURES = [3, 4, 5, 6, 7, 8, 9,
                     16, 17, 18, 19, 20, 21, 22,
                     29, 30, 31, 32, 33, 34, 35,
                     42, 43, 44, 45, 46, 47, 48,
                     55, 56, 57, 58, 59, 60, 61]

# Thresholds for Freezing Index in Hz
LB_LOW = 0.5 # Lower bound of locomotion band index
LB_HIGH = 3 # Upper bound of locomotion band index
FB_LOW = 3 # Lower bound of freeze band index
FB_HIGH = 8 # Upper bound of freeze band index