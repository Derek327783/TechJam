import librosa
import numpy as np
import tensorflow as tf
#import argparse
import json
import tensorflow.keras as keras

# Input dimension for model = 259*13

mappings = ["blues","classical","country","disco","hip-hop","jazz","metal","pop","reggae","rock"]
model_path = "Genre_classifier_model.h5"
class _Genre_spotting_service():
    instance = None
    model = None
    mappings = ["blues", "classical", "country", "disco", "hip-hop", "jazz", "metal", "pop", "reggae", "rock"]

    def __init__(self,model_path):
        self.model = tf.keras.models.load_model(model_path)
    def predict(self,file_path):
        input_data = self.preprocess_audio(file_path)
        input_data = input_data[np.newaxis, ...]
        predictions = self.model.predict(input_data)
        text_predictions = [mappings[np.argmax(predictions, axis=1)[0]]]
        return text_predictions

    def preprocess_audio(self,file_path, target_frames=259, n_mfcc=13, sr=22050, n_fft=2048, hop_length=512):
    # Load audio file
        y, sr = librosa.load(file_path, sr=sr)

    # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Pad or truncate MFCCs to ensure they have target_frames
        if mfccs.shape[1] < target_frames:
            pad_width = target_frames - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :target_frames]
        return mfccs.T


def Genre_spotting_service():
    if _Genre_spotting_service.instance == None:
        _Genre_spotting_service.instance = _Genre_spotting_service()
        _Genre_spotting_service.model = keras.models.load_model(model_path)
    return _Genre_spotting_service.instance

# if __name__ == "__main__":
#     genre_service = _Genre_spotting_service("Genre_classifier_model.h5")
#     predicted_genre = genre_service.predict("10.mp3")
#     print(predicted_genre)


# file_path = "10.mp3"
# target_frames=259
# n_mfcc=13
# sr=22050
# n_fft=2048
# hop_length=512
# # Load audio file
# y, sr = librosa.load(file_path, sr=sr)
#
# # Compute MFCCs
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
# print(str(sr))
#
# # Pad or truncate MFCCs to ensure they have target_frames
# if mfccs.shape[1] < target_frames:
#     pad_width = target_frames - mfccs.shape[1]
#     mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
# else:
#     mfccs = mfccs[:, :target_frames]
# print(mfccs)
