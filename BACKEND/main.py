import librosa  # to extract speech features
import soundfile  # to read audio file
import os
import glob
import pickle  # to save model after training
import numpy as np
from sklearn.model_selection import train_test_split  # for splitting training and testing
from sklearn.neural_network import MLPClassifier  # multi-layer perceptron model
from sklearn.metrics import accuracy_score  # to measure how good we are
from IPython.display import Audio
from playsound import playsound
import warnings

warnings.filterwarnings("ignore")

# All Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

CREMAD_emotions = {
    'SAD': 'sad',
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
}

SAVEE_emotions = {
    'a': 'angry',
    'd': 'disgust',
    'f': 'fearful',
    'h': 'happy',
    'n': 'neutral',
    'sa': 'sad',
    'su': 'surprised',
}

TESS_emotions = {
    'angry.wav': 'angry',
    'disgust.wav': 'disgust',
    'fear.wav': 'fearful',
    'happy.wav': 'happy',
    'neutral.wav': 'neutral',
    'ps.wav': 'surprised',
    'sad.wav': 'sad',
}

# we allow only these emotions lien numero 5
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy",

}


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):  # changing speed
    return librosa.effects.time_stretch(data, rate)


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel, contrast, tonnetz):
    '''
       Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)

    '''

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate  # get the sample rate

        # If chroma is True, get the Short-Time Fourier Transform of X
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))

        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))  # stacking horizontally

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))

        return result


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature_noise(file_name, mfcc, chroma, mel, contrast, tonnetz):
    '''
       Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)

    '''

    with soundfile.SoundFile(file_name) as sound_file:
        simple_audio = sound_file.read(dtype="float32")
        X = noise(simple_audio)
        sample_rate = sound_file.samplerate  # get the sample rate

        # If chroma is True, get the Short-Time Fourier Transform of X
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))

        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))  # stacking horizontally

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))

        return result


# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature_stretch(file_name, mfcc, chroma, mel, contrast, tonnetz):
    '''
       Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)

    '''

    with soundfile.SoundFile(file_name) as sound_file:
        simple_audio = sound_file.read(dtype="float32")
        X = stretch(simple_audio)
        sample_rate = sound_file.samplerate  # get the sample rate

        # If chroma is True, get the Short-Time Fourier Transform of X
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))

        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))  # stacking horizontally

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))

        return result


# Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x, y = [], []

    for file in glob.glob("..\\DATASETS\\RAVDESS\\Actor_*\\*.wav"):
        # get the base name of the audio file
        file_name = os.path.basename(file)
        # get the emotion label
        emotion = emotions[file_name.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        noise = extract_feature_noise(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        stretch = extract_feature_stretch(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)

        # add to data
        x.append(feature)
        y.append(emotion)

        x.append(noise)
        y.append(emotion)

        x.append(stretch)
        y.append(emotion)

    for file in glob.glob("..\\DATASETS\\CREMAD\\*.wav"):
        # get the base name of the audio file
        file_name = os.path.basename(file)
        # get the emotion label
        emotion = CREMAD_emotions[file_name.split("_")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        noise = extract_feature_noise(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        stretch = extract_feature_stretch(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)

        # add to data
        x.append(feature)
        y.append(emotion)

        x.append(noise)
        y.append(emotion)

        x.append(stretch)
        y.append(emotion)

    for file in glob.glob("..\\DATASETS\\SAVEE\\*.wav"):
        # get the base name of the audio file
        file_name = os.path.basename(file)
        part = file_name.split("_")[1]
        emo = part[:-6]

        # get the emotion label
        emotion = SAVEE_emotions[emo]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        noise = extract_feature_noise(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        stretch = extract_feature_stretch(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)

        # add to data
        x.append(feature)
        y.append(emotion)

        x.append(noise)
        y.append(emotion)

        x.append(stretch)
        y.append(emotion)

    for file in glob.glob("..\\DATASETS\\TESS\\*\\*.wav"):
        # get the base name of the audio file
        file_name = os.path.basename(file)
        # get the emotion label
        emotion = TESS_emotions[file_name.split("_")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        noise = extract_feature_noise(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)
        stretch = extract_feature_stretch(file, mfcc=True, chroma=True, mel=True, contrast=False, tonnetz=False)

        # add to data
        x.append(feature)
        y.append(emotion)

        x.append(noise)
        y.append(emotion)

        x.append(stretch)
        y.append(emotion)

    # split the data to training and testing and return it
    return train_test_split(np.array(x), y, test_size=test_size,
                            random_state=7)  # or random start = 7 selon lien 5 ou 9 selon lien numero 1


#  Split the dataset, load RAVDESS dataset, 75% training 25% testing
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# print some details
# Number of samples of the training  data
print("[+] Number of training samples:", x_train.shape[0])
# number of samples of the testing data
print("[+] Number of testing samples:", x_test.shape[0])

# Get the number of features used
# this is a vector of features extracted
# using extract_features() function
print("[+] Number of features:", x_train.shape[1])

# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}

# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(x_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))


# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("..\\result"):
    os.mkdir("..\\result")

pickle.dump(model, open("..\\result/mlp_classifier.model", "wb"))

