import tensorflow.keras as keras
import numpy as np
import librosa
import random
#import tensorflow_addons
SAMPLE_RATE = 22050

class _Emotion_spotting_service():
    model = None
    #instance = None
    mapping = [' amazement', ' solemnity', ' tenderness',
               ' nostalgia', ' calmness', ' power',
               ' joyfulness', ' tension',' sadness']

    def __init__(self,model_path):
        self.model = keras.models.load_model(model_path)
    def predict(self,file_path):
        log_spectrogram = self.preprocess(file_path)
        X = np.array(log_spectrogram).astype("float32")
        X = np.expand_dims(X, axis=0)
        # Do predictions
        num_predictions = self.model.predict(X)
        prediction = np.argmax(num_predictions)
        predicted_keyword = self.mapping[prediction]
        return predicted_keyword

    # Split audio into 10 second excerpts
    # Attain log spectrogram with following parameters
    # sample rate = 22050, n_fft = 2048, hop_length = 512
    # output, 1024*431
    def preprocess(self,file_path):
        signal, sr = librosa.load(file_path,sr=SAMPLE_RATE)
        signal_normalized = librosa.util.normalize(signal)
        len_to_check = 10 * 22050
        # If audio is less than 10 seconds, we pad it with zeroes
        # If audio is more than 10 seconds, we split into segments and randomly choose one
        if len(signal_normalized) < len_to_check:
            num_zeros = len_to_check - len(signal_normalized)
            signal_normalized = signal_normalized + [0] * num_zeros
        elif len(signal_normalized) > len_to_check:
            num_segments = len(signal_normalized)//len_to_check
            segments = []
            for i in range(num_segments):
                start = i * len_to_check
                end = start + len_to_check
                if len(signal[start:end]) != len_to_check:
                    continue
                else:
                    segments.append(signal[start:end])
            signal_normalized = random.choice(segments)
        stft = librosa.stft(signal_normalized, n_fft=2048,hop_length=512)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

# def Emotion_spotting_service():
#     if _Emotion_spotting_service.instance == None:
#         _Emotion_spotting_service.instance = _Emotion_spotting_service()
#         _Emotion_spotting_service.model = keras.models.load_model("ERM.h5")
#     return _Emotion_spotting_service.instance

# if __name__ == "__main__":
#     emotion_service = _Emotion_spotting_service("emotion_model.h5")
#     predicted_word = emotion_service.predict("10.mp3")
#     print(predicted_word)