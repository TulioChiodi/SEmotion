import os
import click
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_io as tfio


@click.command()
@click.argument('csv_filepath', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
def main(csv_filepath, model_path):
    """ Download dataset to data/raw/file_name"""
    PROJECT_PATH = Path(__file__).resolve().parents[2]

    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    @tf.function
    def load_wav_16k_mono(filename):
        """ read in a waveform file and convert to 16 kHz mono """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav


    def load_wav_for_map(filename, label, fold):
        return load_wav_16k_mono(filename), label, fold


    def extract_embedding(wav_data, label, fold):
        ''' run YAMNet to extract embedding from the wav data '''
        _, embeddings, _ = yamnet_model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return (embeddings,
                tf.repeat(label, num_embeddings),
                tf.repeat(fold, num_embeddings))


    class ReduceMeanLayer(tf.keras.layers.Layer):
        def __init__(self, axis=0, **kwargs):
            super(ReduceMeanLayer, self).__init__(**kwargs)
            self.axis = axis

        def call(self, input):
            return tf.math.reduce_mean(input, axis=self.axis)


    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)

    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')

    # CSV_RELATIVE_PATH = 'data/processed/dataset-pt_br/pt_database.csv'
    CSV_PATH = os.path.join(PROJECT_PATH, csv_filepath)

    df = pd.read_csv(CSV_PATH)

    filenames = df['file_fullpath']
    target = df['target']
    folds = df['fold']

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, target, folds))
    main_ds = main_ds.map(load_wav_for_map)
    main_ds = main_ds.map(extract_embedding).unbatch()

    cached_ds = main_ds.cache()
    train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
    test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

    remove_fold_column = lambda embedding, label, fold: (embedding, label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    my_classes = ['anger', 'fear', 'happyness', 'neutral', 'sadness', 'surprise', 'tense']

    my_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), 
                            dtype=tf.float32,
                            name='input_embedding'),
        tf.keras.layers.Dense(512,
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                            activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(my_classes), activation='softmax')
    ], name='my_model')


    my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer="adam",
                    metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=3,
                                                restore_best_weights=True)


    history = my_model.fit(train_ds,
                        epochs=20,
                        validation_data=val_ds,
                        callbacks=callback)


    loss, accuracy = my_model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # MODEL_RELATIVE_PATH = 'model/'
    MODEL_PATH = os.path.join(PROJECT_PATH, model_path)

    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
    embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                                trainable=False, name='yamnet')
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    serving_outputs = my_model(embeddings_output)
    serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
    serving_model = tf.keras.Model(input_segment, serving_outputs)
    serving_model.save(MODEL_PATH, include_optimizer=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()