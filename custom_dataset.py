import csv
import os
import chardet
import shutil
from tqdm import tqdm
import datasets
from datasets import Dataset, load_dataset, Audio, DatasetDict
import numpy as np
import shutil
import soundfile as sf
from pydub import AudioSegment
import librosa
import sys
import yaml
import argparse


script_path = os.path.dirname(__file__).replace("\\", "/")

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

splits = ["train", "validate"]
new_src = os.path.join(script_path, "custom_dataset", cfg.get("augment_type"))

def augment_data(audio, sr):
    ran_aug = np.random.randint(4, size=1)
    if ran_aug== 0:
        # Add random noise
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

def prepare_agument():

    temp_folder = script_path + "/custom_dataset/temp/"

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    counter = 0
    file_names = os.listdir(new_src + "/train")
    for file in tqdm(file_names, total=(len(file_names)), leave=False):

        counter +=1

        audio_path = new_src + "/train/" + file
        audio, sr = librosa.load(audio_path, sr=None)
        audio_files = augment_data(audio, sr)

        wav_file = temp_folder + file.split(".")[0] + ".wav"
        sf.write(wav_file, audio_files, sr)

        # Convert the WAV file to an MP3 file using pydub
        mp3_data = AudioSegment.from_wav(wav_file)
        mp3_data.export(audio_path, format='mp3')

    print("The total number of samples that were augmented is {}".format(counter))
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

def sort():

    training_path = os.path.join(new_src, splits[0])
    validation_path = os.path.join(new_src, splits[1])
    dict_path = os.path.join(new_src, "dict")

    if os.path.exists(training_path):
        shutil.rmtree(training_path)
    if not os.path.exists(training_path):
        os.makedirs(training_path)

    if os.path.exists(validation_path):
        shutil.rmtree(validation_path)
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    if os.path.exists(dict_path):
        shutil.rmtree(dict_path)
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

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
            }
        )

    for split in splits:

        ind = 0
        total = 0
        # Loop through each row in the file
        if split == "train":
            total = cfg.get("train_split")
            dic_path = os.path.join(script_path, cfg.get("dataset_path"), split)
        if split == "validate":
            total = cfg.get("validate_split")
            dic_path = os.path.join(script_path, cfg.get("dataset_path"), split + "d")

        with open(dic_path + '.tsv', 'rb') as d:
                result = chardet.detect(d.read(1000))

            # Open the TSV file in read mode
        with open(dic_path + '.tsv', encoding= result['encoding'],  newline='') as f:
            reader = csv.reader(f, delimiter='\t')

            next(reader)          
                        
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

            #for row in reader:
            for row in tqdm(reader, total=total):
                
                if ind == total:
                    break

                org = os.path.join(script_path, cfg.get("dataset_path"), "clips", row[1])
                path = os.path.join(new_src, split, row[1])

                if not os.path.exists(org):
                    print("file does not exist")
                    continue
                shutil.copyfile(org, path)

                client_id_list.append(row[0])
                path_list.append(path)
                sentence_list.append(row[2])
                up_votes_list.append(row[3])
                down_votes_list.append(row[4])
                gender_list.append(row[6])
                accents_list.append(row[7])
                variant_list.append(row[8])
                locale_list.append(row[9])
                segment_list.append(row[10])

                ind +=1
            
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
                
    if cfg.get("augment_type") == "Base":
        train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
        validate_data = validate_data.cast_column("audio", Audio(sampling_rate=16000))
        custom_dataset = DatasetDict({'train': train_data, 'validate': validate_data})
        custom_dataset.save_to_disk(dict_path)

    elif cfg.get("augment_type") == "Audio-based_Augment":
        print("appling Audio-based augmentation...")
        prepare_agument()
        train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
        validate_data = validate_data.cast_column("audio", Audio(sampling_rate=16000))
        custom_dataset = DatasetDict({'train': train_data, 'validate': validate_data})
        custom_dataset.save_to_disk(dict_path)

def testing():

    test_sample_path = os.path.join(script_path, "custom_dataset/test_samples")
    samples_path = os.path.join(test_sample_path, "samples")
    dict_path = os.path.join(test_sample_path, "dict")

    if os.path.exists(test_sample_path):
        shutil.rmtree(test_sample_path)
    if not os.path.exists(test_sample_path):
        os.makedirs(test_sample_path)

    if os.path.exists(samples_path):
        shutil.rmtree(samples_path)
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    if os.path.exists(dict_path):
        shutil.rmtree(dict_path)
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

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
        }
    )

    dic_path = os.path.join(script_path, cfg.get("dataset_path"), "test.tsv")

    d = open(dic_path, 'rb')
    result = chardet.detect(d.read(1000))

        # Open the TSV file in read mode
    f = open(dic_path, encoding= result['encoding'],  newline='')
    reader = csv.reader(f, delimiter='\t')

    total = -1
       
    for i in reader:
        total +=1

    f.seek(0)
    next(reader)

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

    #for row in reader:
    for row in tqdm(reader, total=total):

        org = os.path.join(script_path, cfg.get("dataset_path"), "clips", row[1])
        path = os.path.join(samples_path, row[1])

        shutil.copyfile(org, path)

        client_id_list.append(row[0])
        path_list.append(path)
        sentence_list.append(row[2])
        up_votes_list.append(row[3])
        down_votes_list.append(row[4])
        gender_list.append(row[6])
        accents_list.append(row[7])
        variant_list.append(row[8])
        locale_list.append(row[9])
        segment_list.append(row[10])

    test_data = Dataset.from_dict({
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

    test_data = test_data.cast_column("audio", Audio(sampling_rate=16000))
    custom_dataset = DatasetDict({'test': test_data})
    custom_dataset.save_to_disk(dict_path)

if cfg.get("create_for_testing"):
    testing()
else:
    sort()

