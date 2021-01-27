# Contains the configuration files for training and dataloader
# Edit the configuration file as per your needs


TRAIN_CSV_PATH = "cut_data.csv"
VALIDATION_CSV_PATH = "test_data_info.csv"
# VALIDATION_CSV_PATH = "test_test_data_info.csv"
TRAIN_IMAGE_DIR = "."
VALIDATION_IMAGE_DIR = "../DATA/tile_round1_testA_20201231/sub_imgs"
TARGET_COL = "target"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TRAIN_WORKERS = 4
LEARNING_RATE = 1e-3
EPOCHS = 100
NUM_CLASSES = 7
DETECTION_THRESHOLD = 0.25

BACKBONE = "resnet50"
MODEL_SAVE_PATH = "models/faster_rcnn_{}".format(BACKBONE)
IS_CONTINUE = False
# valid_batch_size = 4
# valid_workers = 2

BEST_MODEL_PATH = "models/faster_rcnn_resnet50.pth"

OUTPUT_PATH = "outputs/"

PREDICT_IMAGE = None
SAVE_IMAGE = None
SAVE_DIR = "outputs/"
