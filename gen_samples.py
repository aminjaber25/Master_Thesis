import torch
from transformers import pipeline
import os
import datasets
from datasets import load_from_disk, Audio, Dataset, DatasetDict
import random
import logging
import warnings
import tensorflow as tf
from tqdm import tqdm
import shutil
from pydub import AudioSegment
from TTS.api import TTS
from happytransformer import HappyTextToText, TTSettings
import yaml
import argparse
import sys


warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU")
logging.getLogger("happytransformer.happy_transformer").setLevel(logging.WARNING)
tf.get_logger().setLevel(logging.ERROR)
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

path = os.path.join(script_path, cfg.get("dataset"))
data = load_from_disk(path)

# check gpu/cpu and load model
mask_model_path = os.path.join(script_path, cfg.get("mask_model_path"))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
unmasker = pipeline('fill-mask',
                    model= mask_model_path,
                    tokenizer=mask_model_path,
                    device=device)

# Init TTS with the target model name
tts = TTS(model_name=cfg.get("TTS_model_name"), progress_bar = False, gpu = cfg.get("TTS_use_gpu"))

# Initiate grammer corrector model
happy_tt = HappyTextToText("T5", cfg.get("grammer_model_name"))
beam_settings = TTSettings(num_beams=5, min_length=1)


def random_arranged_list(length):

    numbers = list(range(length))
    random_arranged = random.sample(numbers, len(numbers))

    return random_arranged

def get_unmask_text(temp_text, pre_num = 1):

    return unmasker(temp_text, top_k = pre_num)

def check_alphabetic(word):
    for char in word:
        if not char.isalpha():
            return True
        else:
            return False
        
# run the mask model
def generate_text(text, pre_num = 1):

    words = text.split()

    if len(words) <= 3:
        return text
    
    else:
        word_masking_ratio = cfg.get("word_masking_ratio")
        mask_word_idx = random_arranged_list(len(words))
        mask_word_idx = mask_word_idx[:int(len(mask_word_idx) * word_masking_ratio)]
        mask = "[MASK]"
        gen_text = ""
        

        for masking in mask_word_idx:

            masked_word = words[masking]
            words[masking] = mask
            temp_text = " ".join(words)

            gen_text = get_unmask_text(temp_text)
            count = 1
            while(check_alphabetic(gen_text[count - 1]["token_str"])):
                if count <= 3:
                    count += 1
                    gen_text = get_unmask_text(temp_text, count)
                else:
                    break
                    
            if count == 4:
                words[masking] = masked_word
            else:
                words[masking] = (gen_text[count - 1]["token_str"])

        sen = " ".join(words)

        return sen

# generate audio from the new text
def generate_audio(text, gen_name_path):

    tts.tts_to_file(text= text, file_path= gen_name_path)

    mp3_data = AudioSegment.from_wav(gen_name_path)
    mp3_path = gen_name_path.rsplit(".",1)[0] + ".mp3"
    mp3_data.export(mp3_path, format='mp3')
    os.remove(gen_name_path)

    return mp3_path

gen_dataset_path = os.path.join(script_path, "custom_dataset", cfg.get("gen_dataset_name"))
train_dataset_path = os.path.join(gen_dataset_path, "train")
validate_dataset_path = os.path.join(gen_dataset_path, "validate")
dict_dataset_path = os.path.join(gen_dataset_path, "dict")

if os.path.exists(gen_dataset_path):
    shutil.rmtree(gen_dataset_path)
if not os.path.exists(gen_dataset_path):
    os.makedirs(gen_dataset_path)

if not os.path.exists(train_dataset_path):
    os.makedirs(train_dataset_path)

if not os.path.exists(validate_dataset_path):
    os.makedirs(validate_dataset_path)

if not os.path.exists(dict_dataset_path):
    os.makedirs(dict_dataset_path)

write_train_file = open(os.path.join(gen_dataset_path, "train_sentences.text"), "w", encoding="utf-8")
write_validate_file = open(os.path.join(gen_dataset_path, "validation_sentences.text"), "w", encoding="utf-8")

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


for i in tqdm(range(len(data["train"])), total = len(data["train"])):
    
    sample = data["train"][i]

    gen_name_path = os.path.join(train_dataset_path, (sample["audio"]["path"].rsplit(".",1)[0] + "_gen.wav"))
    new_text = ""
    # clean text from any  punctuation marks
    for char in sample["sentence"]:
        if char in [",", ";", ".", ":", "„", "“", "\"", "!", "?", "-", "_"]:
            continue
        else:
            new_text = new_text + char
    # start generating new sentences
    gen_sentence = generate_text(new_text)
    
    # check the new sentences
    sentence = happy_tt.generate_text("grammar: " + gen_sentence, args=beam_settings).text

    # wite the new generated output in a txt file
    write_train_file.write("original sentence: {}".format(new_text) + "\n")
    write_train_file.write("generated sentence: {}".format(gen_sentence) + "\n")
    write_train_file.write("corrected sentence: {}".format(sentence) + "\n")

    # generate the audio file
    mp3_sample_path = generate_audio(sentence, gen_name_path)
    
    client_id_list.append(sample["client_id"])
    path_list.append(mp3_sample_path)
    sentence_list.append(sentence)
    up_votes_list.append(sample["up_votes"])
    down_votes_list.append(sample["down_votes"])
    gender_list.append(sample["gender"])
    accents_list.append(sample["accents"])
    variant_list.append(sample["variant"])
    locale_list.append(sample["locale"])
    segment_list.append(sample["segment"])
    

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

write_train_file.close()

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

for i in tqdm(range(len(data["validate"])), total = len(data["validate"])):

    sample = data["validate"][i]
    org = sample["path"]
    dist = os.path.join(validate_dataset_path, sample["audio"]["path"])
    shutil.copyfile(org, dist)

    client_id_list.append(sample["client_id"])
    path_list.append(dist)
    sentence_list.append(sample["sentence"])
    up_votes_list.append(sample["up_votes"])
    down_votes_list.append(sample["down_votes"])
    gender_list.append(sample["gender"])
    accents_list.append(sample["accents"])
    variant_list.append(sample["variant"])
    locale_list.append(sample["locale"])
    segment_list.append(sample["segment"])
    

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

write_validate_file.close()

train_data = train_data.cast_column("audio", Audio(sampling_rate=16000))
validate_data = validate_data.cast_column("audio", Audio(sampling_rate=16000))
custom_dataset = DatasetDict({'train': train_data, 'validate': validate_data})
custom_dataset.save_to_disk(dict_dataset_path)