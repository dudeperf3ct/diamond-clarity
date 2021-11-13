import os
import pickle
from PIL import Image
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from models import cnn3d
from vis import plot_cm, plot_curves
import wandb
from wandb.keras import WandbCallback

epochs = 15
batch_size = 1
split_ratio = 0.05
lr = 0.0001
height = width = 384
depth = 6
channels = 3
random_seed = 42
use_wandb = True


def train_preprocessing(images):
    images = images - [127.5]
    images = images / 128.
    return images

def valid_preprocessing(images):
    images = images - [127.5]
    images = images / 128.
    return images


class ClarityClassifier:

    def __init__(self, experiment, dataset_path, root_dir, model_name, model_dir) -> None:
        self.dataset_path = dataset_path
        self.root_dir = root_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.get_model(model_name)

        # model params
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=experiment)

    def get_model(self, model_name):
        if model_name == 'cnn3d':
            self.model = cnn3d.cnn3d(width, height, depth, channels)
        self.model.summary()

    def load_dataset(self):
        with open(self.dataset_path, mode='rb') as f:
            ds = pickle.load(f)
        ds = ds.reset_index()
        ds = ds[ds['er.Mike'].notna()]
        ds.loc[ds['er.Mike'] == 0.5, 'er.Mike'] = 0
        ds['er.Mike'] = ds['er.Mike'].astype('string')
        data = ds[['SKU', 'er.Mike']]
        return pd.DataFrame(data.values, columns=["fname", "label"])

    def generator_fn(self, img_filepath, lbls=None, is_valid=False):
        def generator():
            for idx in range(len(img_filepath)):
                images = []
                for i in range(6):
                    if self.root_dir is not None:
                        img_filename = os.path.join(self.root_dir, img_filepath[idx]+f'_{i}.jpg')
                    else:
                        img_filename = img_filepath[idx]
                    img = np.array(Image.open(img_filename).convert("RGB"))
                    if is_valid:
                        img = valid_preprocessing(img)
                    else:
                        img = train_preprocessing(img)
                    images.append(img)
                # shuffle(images)
                if lbls is not None:
                    # N,C​,D,H,W
                    yield np.stack(images, -1).transpose(0,1,3,2), lbls[idx]
                else:
                    yield images
        return generator

    def _prepare_training_generators(self, df: pd.DataFrame) -> tuple:
        """Prepare training and validation dataloaders"""
        x = df["fname"]
        y = df["label"]

        train_x, val_x, train_y, val_y = train_test_split(
            x.values,
            y.values,
            test_size=self.split_ratio,
            random_state=self.random_seed,
            shuffle=True,
            stratify=y
        )
        # check if model is overfitting
        # train_x, val_x, train_y, val_y = train_x[:3], val_x[:1], train_y[:3], val_y[:1]
        print(
            f"Training shape: {train_x.shape}, {train_y.shape}, {np.unique(train_y, return_counts=True)}"
        )
        print(
            f"Validation shape: {val_x.shape}, {val_y.shape}, {np.unique(val_y, return_counts=True)}"
        )
        # define dataloaders
        train_gen = self.generator_fn(train_x, train_y)
        train_loader = tf.data.Dataset.from_generator(
                        generator=train_gen, 
                        output_types=(np.float32, np.int32), 
                        output_shapes=((width, height, depth, channels), ())
                    )
        valid_gen = self.generator_fn(val_x, val_y, True)
        validation_loader = tf.data.Dataset.from_generator(
                        generator=valid_gen,
                        output_types=(np.float32, np.int32), 
                        output_shapes=((width, height, depth, channels), ())
                    )
        # Augment the on the fly during training.
        train_dataset = (
            train_loader.shuffle(len(train_x), seed=self.random_seed)
            # .map(train_preprocessing, num_parallel_calls=AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(val_x), seed=self.random_seed)
            # .map(validation_preprocessing, num_parallel_calls=AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return train_dataset, validation_dataset

    def create_dataset(self):
        
        ds = self.load_dataset()
        lbl = LabelEncoder()
        y = lbl.fit_transform(ds["label"])
        self.classes = lbl.classes_
        print("Label Encoding: {}".format(lbl.classes_))
        self.class_mapping = {k: v for k, v in enumerate(lbl.classes_)}
        print('Label Mapping: {}'.format(self.class_mapping))
        ds['label'] = y
        with open(self.model_dir + "/class_mapping.json", "w") as fp:
            json.dump(self.class_mapping, fp)
        cws = class_weight.compute_class_weight("balanced", classes=np.unique(ds["label"]), y=ds["label"])
        print(f"Class weights for labels: {cws}")

        print("Creating dataset")
        train_dataset, validation_dataset = self._prepare_training_generators(ds)
        return train_dataset, validation_dataset

    def train(self, train_dataset, validation_dataset):

        # Compile model.
        initial_learning_rate = self.lr
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=10, decay_rate=0.96, staircase=True
        )
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1, axis=-1, name='binary_crossentropy')

        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["acc", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        # Define callbacks.
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            self.model_dir + f"/{self.model_name}.h5", save_best_only=True
        )
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

        callbacks = [checkpoint_cb, early_stopping_cb]
        if use_wandb:
            callbacks.append(WandbCallback())

        # Train the model, doing validation at the end of each epoch
        self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.epochs,
            shuffle=True,
            verbose=2,
            callbacks=callbacks,
        )

        fig = plot_curves(self.model)
        plt.show()
        if self.use_wandb:
            wandb.log({"img": [wandb.Image(fig, caption=f"{self.model_name}_curves")]})

    def evaluate(self, dataset):
        y_true, y_pred = [], []

        for image_batch, label_batch in dataset:
            y_true.append(label_batch)
            preds = self.model.predict(image_batch)
            y_pred.append((preds > 0.5).astype('int32'))

        # convert the true and predicted labels into tensors
        correct_labels = tf.concat([item for item in y_true], axis = 0)
        predicted_labels = tf.concat([item for item in y_pred], axis = 0)

        print(classification_report(correct_labels, predicted_labels, target_names=self.classes))

        fig = plot_cm(correct_labels, predicted_labels, self.classes)
        plt.show(fig)
        if self.use_wandb:
            wandb.log({"img": [wandb.Image(fig, caption=f"{self.model_name}_confusion_matrix")]})
