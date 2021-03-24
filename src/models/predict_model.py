import click
import logging
from pathlib import Path
import pyaudio
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os
import hyper

@click.command()
@click.argument('model_path', type=click.Path())
def main(model_path):
    """  Run the predictions and send them to Hyperorganicos server
         via MQTT protocol
    """
    LOG_FILEPATH = os.path.join(PROJECT_PATH, model_path, 'EPredictiong.log')
    LITE_LOG_FILEPATH = os.path.join(PROJECT_PATH, model_path, 'Lite_EPredictiong.log')

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
    def load_wav_16k_mono(data):
        """ read in a waveform file and convert to 16 kHz mono """
        sample_rate=44100
        data_tf = tf.convert_to_tensor(data, np.float32)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(data_tf, rate_in=sample_rate, rate_out=16000)
        return wav

    saved_model_path = os.path.join(PROJECT_PATH, model_path)
    model = tf.saved_model.load(saved_model_path)

    #  Time streaming #############################################
    RATE = 44100 # Sample rate
    nn_time = 3 # signal length send to the network
    CHUNK = round(RATE*nn_time) # Frame size

    # Identificar dispositivos de audio do sistema
    # p = pyaudio.PyAudio()
    # info = p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')
    # for i in range(0, numdevices):
    #         if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #             print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    #             if p.get_device_info_by_host_api_device_index(0, i).get('name') == 'default':
    #                 print(f'The correct device id is {i}')

    print('janela de análise é de: {0} segundos'.format(CHUNK/RATE))
    # input stream setup
    # pyaudio.paInt16 : representa resolução em 16bit
    
    p = pyaudio.PyAudio()

    stream=p.open(format = pyaudio.paFloat32,
                        rate=RATE,
                        channels=1, 
                        input=True,  
                        frames_per_buffer=CHUNK)


    # labels = ['Irritação', 'Aversão', 'Medo', 'Alegria', 'Neutro', 'Tristeza', 'Surpresa']
    my_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

    # Connecting to Hyperorganicos MQTT broker 
    hyper.connect()

    topic_pub = 'hiper/labinter0'

    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        sample = load_wav_16k_mono(data)
        final_layer = model(sample)
        final_layer_np = final_layer.numpy()
        pred_idx = tf.argmax(final_layer)
        pred = my_classes[pred_idx]
        logger.debug(final_layer_np)
        logger.info(f'The main emotion is: {pred}')
        hyper.send(topic_pub='hiper/labinter99', message=pred, output=False)
        for counter, item in enumerate(final_layer_np):
            topic_pub_lane = ''.join([topic_pub, str(counter)])
            hyper.send(topic_pub=topic_pub_lane, message=str(item), output=False)

    hyper.disconnect()

if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    PROJECT_PATH = Path(__file__).resolve().parents[2]

    main()
