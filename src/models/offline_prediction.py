import click
import logging
from pathlib import Path
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os

@click.command()
@click.argument('model_path', type=click.Path())
@click.argument('audio_filepath', type=click.Path())
def main(model_path, audio_filepath):
    """  Run the predictions and send them to Hyperorganicos server
         via MQTT protocol
    """
    LOG_FILEPATH = os.path.join(PROJECT_PATH, model_path, 'OfflinePredictiong.log')
    LITE_LOG_FILEPATH = os.path.join(PROJECT_PATH, model_path, 'Lite_OfflineEPredictiong.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(LOG_FILEPATH)
    file_handler.setFormatter(formatter)

    lite_file_handler = logging.FileHandler(LITE_LOG_FILEPATH)
    lite_file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    
    logger.addHandler(file_handler)
    logger.addHandler(lite_file_handler)
    logger.addHandler(stream_handler)

    logger.debug('running prediction mode')

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

    saved_model_path = os.path.join(PROJECT_PATH, model_path)
    model = tf.saved_model.load(saved_model_path)

    my_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

    sample = load_wav_16k_mono(audio_filepath)
    final_layer = model(sample)
    final_layer_np = final_layer.numpy()
    pred_idx = tf.argmax(final_layer)
    pred = my_classes[pred_idx]
    logger.debug(final_layer_np)
    logger.info(f'The main emotion is: {pred}')


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    PROJECT_PATH = Path(__file__).resolve().parents[2]

    main()
