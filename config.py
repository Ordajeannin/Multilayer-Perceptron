LEARNING_RATE = 0.01
EPOCHS = 50
HIDDEN_SIZES = [32, 32]
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
CV_FOLDS = 5
CV_SEED = 42


RUN_NAME = "32_32_lr001"

DATASET_PATH = "source/data.csv"
LOSS_PLOT_PATH = f"files/{RUN_NAME}/loss_{RUN_NAME}.png"
ACCURACY_PLOT_PATH = f"files/{RUN_NAME}/accuracy_{RUN_NAME}.png"
HISTORY_PATH = f"files/{RUN_NAME}/history_{RUN_NAME}.json"
MODEL_PATH = f"model/model_{RUN_NAME}.json"
METRICS_PATH = f"files/{RUN_NAME}/metrics_{RUN_NAME}.json"
CV_RESULTS_PATH = f"files/{RUN_NAME}/cv_results_{RUN_NAME}.json"