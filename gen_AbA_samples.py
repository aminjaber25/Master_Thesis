import os
import shutil
from tqdm import tqdm
import datasets
from datasets import DatasetDict, Dataset, Audio, load_from_disk
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa
from scipy import signal
import random
from scipy.signal import convolve
import yaml
import argparse
import sys


script_path = os.path.dirname(__file__)
parser = argparse.ArgumentParser(description='Process a configuration file.')
parser.add_argument('config_path', nargs='?', default= os.path.join(script_path, "config.yaml"), help='Path to the configuration file (e.g., config.yaml)')
args = parser.parse_args()

# loading config file
def load_yaml_config(config_path):

    try:
        file = open(config_path, "r")
        config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found.")
        sys.exit(1)

cfg = load_yaml_config(args.config_path)


def echo(audio, sample_rate):
    
    # sample_rate, audio = wavfile.read(wave_file)

    # Define room echo parameters
    delay = 0.25  # Delay time in seconds
    attenuation = round(random.uniform(0.2, 0.3), 2)  # Attenuation factor 0.2 - 0.3

    # Create the echo effect using FIR filter
    num_samples_delay = int(delay * sample_rate)
    echo_filter = signal.lfilter([1], [1] + [0] * num_samples_delay + [attenuation], audio)

    # Normalize the output
    echo_filter = echo_filter / max(echo_filter)

    return echo_filter
    # wavfile.write(wave_file, sample_rate, echo_filter)

def reverb(audio, sample_rate):

    # Load the original audio
    # sample_rate, audio_data = wavfile.read(wave_file)

    # Create the reverb impulse response
    reverb_duration = round(random.uniform(0.1, 0.3), 2)  # Duration of the reverb in seconds 0.1 - 0.3

    reverb_strength = 0.4  # Adjust this value to control reverb strength
    reverb_sample_count = int(sample_rate * reverb_duration)

    # Create a shorter decay impulse response
    decay_impulse_response = np.exp(-np.arange(reverb_sample_count) / (sample_rate * reverb_strength))

    reverberated_audio = convolve(audio, decay_impulse_response, mode='full')

    # Normalize the audio to avoid clipping
    reverberated_audio = np.int16(reverberated_audio / np.max(np.abs(reverberated_audio)) * 32767)

    return reverberated_audio
    # wavfile.write(wave_file, sample_rate, reverberated_audio)

def background_noise(audio, sample_rate):

    bg_samples_path = os.path.join(script_path, cfg.get("Background_Noise_path"))
    if not os.path.exists(bg_samples_path):
        print("there folder {} does not exist".format(cfg.get("Background_Noise_path")))
        sys.exit(1)

    list_bg_files = os.listdir(bg_samples_path)
    random_integer = random.randint(0, len(list_bg_files)-1)
    bg_file = os.path.join(bg_samples_path, list_bg_files[random_integer])

    background_audio, background_sr = librosa.load(bg_file, sr=sample_rate)

    num_repeats = len(audio) / len(background_audio)
    background_audio = background_audio * num_repeats
    repeated_background = background_audio[:len(audio)]
    background_volume = 0.7
    mixed_audio =  background_volume * repeated_background + audio

    return mixed_audio

def augment_data_speaker(audio, sr):
    ran_aug = np.random.randint(7, size=1)
    if ran_aug== 0:
        # Add random white noise
        noise = np.random.randn(len(audio))
        audio_noise = audio + 0.005 * noise
        return audio_noise
    if ran_aug== 1:
        # Time stretching
        rate = np.random.uniform(0.8, 1.2)
        audio_stretch = librosa.effects.time_stretch(audio, rate=rate)
        return audio_stretch
    if ran_aug== 2:
        # Pitch shifting
        n_steps = np.random.randint(-3, 4)
        audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return audio_pitch
    if ran_aug == 3:
        # Time shifting
        shift_range = 0.5  # Maximum shift range in seconds
        # Randomly generate a shift amount within the given range
        shift_amount = np.random.uniform(low=-shift_range, high=shift_range)
        # Compute the shift amount in samples
        shift_samples = int(shift_amount * sr)
        audio_shift = np.roll(audio, shift_samples)
        return audio_shift
    elif ran_aug == 4:
        return echo(audio, sr)
    elif ran_aug == 5:
        return reverb(audio, sr)
    elif ran_aug == 6:
        return background_noise(audio, sr)

def prepare_agument():

    temp_folder = os.path.join(script_path, "custom_dataset", "temp")
    gen_classic_dataset_path = os.path.join(script_path, "custom_dataset", cfg.get("gen_AbA_dataset_name"))
    train_dataset_path = os.path.join(gen_classic_dataset_path, "train")
    validate_dataset_path = os.path.join(gen_classic_dataset_path, "validate")
    dict_dataset_path = os.path.join(gen_classic_dataset_path, "dict")

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    if os.path.exists(gen_classic_dataset_path):
        shutil.rmtree(gen_classic_dataset_path)
    if not os.path.exists(gen_classic_dataset_path):
        os.makedirs(gen_classic_dataset_path)

    if not os.path.exists(train_dataset_path):
        os.makedirs(train_dataset_path)
    
    if not os.path.exists(validate_dataset_path):
        os.makedirs(validate_dataset_path)

    if not os.path.exists(dict_dataset_path):
        os.makedirs(dict_dataset_path)

    splits = ["train", "validate"]
    dataset_path = os.path.join(script_path, "custom_dataset", cfg.get("gen_Base_dataset_name"))
    train_path = os.path.join(dataset_path, splits[0])
    validate_path = os.path.join(dataset_path, splits[1])
    train_files = os.listdir(train_path)
    validate_files = os.listdir(validate_path)

    data_path = os.path.join(dataset_path, "dict")
    data = load_from_disk(data_path)

    for samples_no in tqdm(range(len(data["train"]))):

        sample_name = data["train"][samples_no]["path"].rsplit("\\",1)[1]
        audio, sr = librosa.load(data["train"][samples_no]["path"], sr=None)
        audio_files = augment_data_speaker(audio, sr)

        # bg_audio = augment_data_Background(audio_files, sr)
        
        wav_file = os.path.join(temp_folder, sample_name.split(".")[0] + ".wav")
        sf.write(wav_file, audio_files, sr)

        

        # Convert the WAV file to an MP3 file using pydub
        mp3_data = AudioSegment.from_wav(wav_file)
        mp3_data.export(os.path.join(train_dataset_path, sample_name), format='mp3')
        # subprocess.run(['ffmpeg', '-i', wav_file, os.path.join(train_dataset_path, sample_name)])

    for samples in tqdm(validate_files, total= len(validate_files)):

        shutil.copyfile(os.path.join(validate_path, samples),os.path.join(validate_dataset_path, samples))

    features = datasets.Features(
        {
            "client_id": datasets.Value("string"),
            "path": datasets.Value("string"),
            "audio": datasets.features.Audio(sampling_rate=48_000),
            "sentence": datasets.Value("string", id=None),
            "up_votes": datasets.Value("int64"),
            "down_votes": datasets.Value("int64"),
            "gender": datasets.Value("string"),
            "accents": datasets.Value("string"),
            "variant": datasets.Value("string"),
            "locale": datasets.Value("string"),
            "segment": datasets.Value("string"),
        })


    for split in splits:

        client_id_list =[]
        path_list =[]
        sentence_list =[]
        up_votes_list =[]
        down_votes_list =[]
        gender_list =[]
        accents_list =[]
        variant_list =[]
        locale_list =[]
        segment_list =[]

        for i in tqdm(range(len(data[split]))):
            
            samples = data[split][i]

            client_id_list.append(samples["client_id"])
            if split=="train":
                path_list.append(os.path.join(train_dataset_path, samples["path"].rsplit("\\",1)[1]))
            else:
                path_list.append(os.path.join(validate_dataset_path, samples["path"].rsplit("\\",1)[1]))
            sentence_list.append(samples["sentence"])
            up_votes_list.append(samples["up_votes"])
            down_votes_list.append(samples["down_votes"])
            gender_list.append(samples["gender"])
            accents_list.append(samples["accents"])
            variant_list.append(samples["variant"])
            locale_list.append(samples["locale"])
            segment_list.append(samples["segment"])
            
        if split=="train":
            train_data = Dataset.from_dict({
                "client_id": client_id_list,
                "path": path_list,
                "audio": path_list,
                "sentence": sentence_list,
                "up_votes": up_votes_list,
                "down_votes": down_votes_list,
                "gender": gender_list,
                "accents": accents_list,
                "variant": variant_list,
                "locale": locale_list,
                "segment": segment_list,
                }, features=features)
        else:
            validate_data = Dataset.from_dict({
                "client_id": client_id_list,
                "path": path_list,
                "audio": path_list,
                "sentence": sentence_list,
                "up_votes": up_votes_list,
                "down_votes": down_votes_list,
                "gender": gender_list,
                "accents": accents_list,
                "variant": variant_list,
                "locale": locale_list,
                "segment": segment_list,
                }, features=features)
                

    train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
    validate_data = validate_data.cast_column("audio", Audio(sampling_rate=16000))
    custom_dataset = DatasetDict({'train': train_data, 'validate': validate_data})
    custom_dataset.save_to_disk(dict_dataset_path)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

prepare_agument()

