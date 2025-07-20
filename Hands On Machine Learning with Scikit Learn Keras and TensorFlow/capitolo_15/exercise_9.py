# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:35:11 2025

@author: trotta

9.	Eseguire il training di un modello di classificazione per il set di dati SketchRNN, disponibile nei set di dati TensorFlow.

"""
from tensorflow import keras
DOWNLOAD_ROOT = "http://download.tensorflow.org/data/"
FILENAME = "quickdraw_tutorial_dataset_v1.tar.gz"
filepath = keras.utils.get_file(FILENAME, DOWNLOAD_ROOT + FILENAME, cache_subdir="datasets/quickdraw", extract=True)
# %%
from pathlib import Path
quickdraw_dir = Path(filepath).parent
train_files = sorted([str(path) for path in quickdraw_dir.glob("training.tfrecord-*")])
eval_files = sorted([str(path) for path in quickdraw_dir.glob("eval.tfrecord-*")])
# %%
with open(quickdraw_dir / "eval.tfrecord.classes") as test_classes_file:
    test_classes = test_classes_file.readlines()
    
with open(quickdraw_dir / "training.tfrecord.classes") as train_classes_file:
    train_classes = train_classes_file.readlines()
