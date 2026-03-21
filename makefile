PYTHON := python3

DATASET := source/data.csv
TRAIN_DATA := data/train.csv
VALID_DATA := data/valid.csv

CURRENT_MODEL := model/model_16_16_lr001.json

.PHONY: all help init split train predict cv compare-shape compare-lr compare-metrics compare-cv clean fclean re

all: train

help:
	@echo "Available targets:"
	@echo "  make init             - create package __init__.py files"
	@echo "  make split            - split source/data.csv into data/train.csv and data/valid.csv"
	@echo "  make train            - train the current model from config/config.py"
	@echo "  make predict          - run prediction on validation set"
	@echo "  make cv               - run cross-validation with current config"
	@echo "  make compare-shape    - compare architectures with learning curves"
	@echo "  make compare-lr       - compare learning rates with learning curves"
	@echo "  make compare-metrics  - compare metrics.json files"
	@echo "  make compare-cv       - rank models using cv_results files"
	@echo "  make clean            - remove python cache files"
	@echo "  make fclean           - remove generated files/models/visualizations"
	@echo "  make re               - fclean then train"

init:
	touch script/__init__.py
	touch config/__init__.py
	touch utils/__init__.py
	touch compare/__init__.py

split: init
	mkdir -p data
	$(PYTHON) -m script.split $(DATASET) $(TRAIN_DATA) $(VALID_DATA)

train: init
	mkdir -p data model files visualizations
	$(PYTHON) -m script.train $(TRAIN_DATA) $(VALID_DATA)

predict: init
	$(PYTHON) -m script.predict $(CURRENT_MODEL) $(VALID_DATA)

cv: init
	mkdir -p files
	$(PYTHON) -m script.cross_validation $(DATASET)

compare-shape: init
	mkdir -p files/compare_shape
	$(PYTHON) -m compare.compare_histories \
		8-8=files/8_8_lr001/history_8_8_lr001.json \
		16-8=files/16_8_lr001/history_16_8_lr001.json \
		16-16=files/16_16_lr001/history_16_16_lr001.json \
		32-16=files/32_16_lr001/history_32_16_lr001.json \
		32-32=files/32_32_lr001/history_32_32_lr001.json

compare-lr: init
	mkdir -p files/compare_lr
	$(PYTHON) -m compare.compare_histories \
		16-16-lr0.1=files/16_16_lr01/history_16_16_lr01.json \
		16-16-lr0.01=files/16_16_lr001/history_16_16_lr001.json \
		16-16-lr0.001=files/16_16_lr0001/history_16_16_lr0001.json

compare-metrics: init
	$(PYTHON) -m compare.compare_metrics \
		8-8=files/8_8_lr001/metrics_8_8_lr001.json \
		16-8=files/16_8_lr001/metrics_16_8_lr001.json \
		16-16=files/16_16_lr001/metrics_16_16_lr001.json \
		32-16=files/32_16_lr001/metrics_32_16_lr001.json \
		32-32=files/32_32_lr001/metrics_32_32_lr001.json

compare-cv: init
	$(PYTHON) -m compare.compare_cv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

fclean: clean
	rm -rf files/*
	rm -rf model/*
	rm -rf visualizations/*

re: fclean train