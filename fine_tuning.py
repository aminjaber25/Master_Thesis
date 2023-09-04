import sys
from datasets import load_dataset, load_from_disk, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from transformers import Seq2SeqTrainingArguments, TrainerControl, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import os
import numpy as np
import random
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

model_name = cfg.get("model_name")


if cfg.get("from_checkpoint"):
    out_put_model = model_path = os.path.join(script_path, model_name)
elif cfg.get("from_scratch"):
    model_path = "openai/whisper-" + cfg.get("from_scratch_model")
    out_put_model = os.path.join(script_path, model_name)
elif cfg.get("fine_tuning"):
    model_path = os.path.join(script_path, model_name)
    out_put_model = os.path.join(script_path, cfg.get("augmentations"))
else:
    print("Please choose one of the options available.")
    sys.exit(1)

path = os.path.join(script_path, "custom_dataset", cfg.get("dataset_cycle"), "dict")
data = load_from_disk(path)

def grad_cal(batch_size):

    if batch_size <=16:
        return int(16/batch_size)
    else: return 1

epoch_per_cycle = cfg.get("epoch")
num_train_steps = int(len(data["train"])/(cfg.get("batch_size")*2))
gradient_steps = grad_cal(cfg.get("batch_size"))
num_val_save_steps = int(num_train_steps/5)
num_of_max_steps= (num_train_steps * epoch_per_cycle)

data = data.remove_columns(["accents", "variant", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]) # for custom dataset

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path, language="German", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_path, language="German", task="transcribe")

data = data.cast_column("audio", Audio(sampling_rate=16000))

def SpecAugment(spec, num_freq_mask, freq_percentage, num_time_mask, time_percentage):
    f0 = 0
    # get the number of frames and the number of frequencies.
    all_frames_num, all_freqs_num = spec.shape

    # defines the amount of masking given a percentage.
    num_freqs_to_mask = int(freq_percentage * all_freqs_num)
    freq_chunk_size = int(all_freqs_num/num_freq_mask)

    # defines the amount of masking given a percentage.
    num_frames_to_mask = int(time_percentage * all_frames_num)
    frame_chunk_size = int(all_frames_num/num_time_mask)

    for i in range(num_freq_mask):

        # defines which frequency will be masked.
        f0 = np.random.randint(freq_chunk_size * i, freq_chunk_size * (i+1))

        # masks the frequency by assigning zero.
        spec[:, f0:f0 + num_freqs_to_mask] = 0

    t0=0
    for j in range(num_time_mask):

        # defines which instant of time will be masked.
        t0 = np.random.randint(frame_chunk_size * j, frame_chunk_size * (j+1))

        # masks the instant of time by assigning zero.
        spec[t0:t0 + num_frames_to_mask, :] = 0 

        
    return spec

def MixSpeech(spec_1, spec_2):
    
    # Set the opacity values and blend the images using opacity values
    opacity_base = 0.8
    opacity_overlay = 0.2

    width = 0

    # for w in range(spec_1.shape[1]+1):
    for h in range(spec_1.shape[0]):    
        for w in range(spec_1.shape[1]):
            if spec_1[h:h+1,w:w+1]> 0 :
                width = w

    temp = spec_1[:,:width+1] * opacity_base + spec_2[:,:width+1] + opacity_overlay
    blended_array = np.concatenate((temp, spec_1[:,temp.shape[1]:]), axis=1)

    return blended_array

def SpecSwap(spec, frame_chunk_size= 0.3, freq_chunk_size= 0.3):
    
    # f0 = 0
    # get the number of frames and the number of frequencies.
    all_frames_num, all_freqs_num = spec.shape

    spec_copy = np.zeros(spec.shape)

    freqs_window = 0
    
    # last freq element to avoid the padding
    for i in range(all_freqs_num):
        if spec[0,i] > 0:
            freqs_window = i

    half_freqs_window = int(freqs_window/2)
    
    # first half
    freqst_a = random.randint(0, int(half_freqs_window * (1-freq_chunk_size)))
    freqend_a = int(freqst_a + (half_freqs_window * freq_chunk_size))

    spec_copy[:, freqst_a:freqend_a] = spec[:, freqst_a:freqend_a]

    #second half
    freqst_b = random.randint(half_freqs_window, int(freqs_window * (1-freq_chunk_size)))
    freqend_b = int(freqst_b + (half_freqs_window * freq_chunk_size))

    spec_copy[:, freqst_b:freqend_b] = spec[:, freqst_b:freqend_b]

    spec[:, freqst_a:freqend_a] = spec_copy[:, freqst_b:freqend_b]
    spec[:, freqst_b:freqend_b] = spec_copy[:, freqst_a:freqend_a]


    half_fram_window = int(all_frames_num/2)

    # first half
    framst_a = random.randint(0, int(half_fram_window * (1-frame_chunk_size)))
    framend_a = int(framst_a + (half_fram_window * frame_chunk_size))

    spec_copy[framst_a:framend_a, :] = spec[framst_a:framend_a, :]

    #second half
    framst_b = random.randint(half_fram_window, int(all_frames_num * (1-frame_chunk_size)))
    framend_b = int(framst_b + (half_fram_window * frame_chunk_size))

    spec_copy[framst_b:framend_b, :] = spec[framst_b:framend_b, :]

    spec[framst_a:framend_a, :] = spec_copy[framst_b:framend_b, :]
    spec[framst_b:framend_b, :] = spec_copy[framst_a:framend_a, :]
        
    return spec

def prepare_dataset(batch):

    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def prepare_dataset_SpecAugment(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    freq_masking_min_percentage = 0.002
    freq_masking_max_percentage = 0.010
    num_freq_mask = 16

    freq_percentage = random.uniform(freq_masking_min_percentage, freq_masking_max_percentage)

    time_masking_min_percentage = 0.02
    time_masking_max_percentage = 0.03
    num_time_mask = 4

    time_percentage = random.uniform(time_masking_min_percentage, time_masking_max_percentage)

    # plot_spect(SpecAugment(batch["input_features"], num_freq_mask, freq_percentage, num_time_mask, time_percentage), title="Mel Spectrogram", ylabel="Mel Filter", aspect="auto", xmax=None)
    batch["input_features"] = SpecAugment(batch["input_features"], num_freq_mask, freq_percentage, num_time_mask, time_percentage)
        
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch

def prepare_dataset_MixSpeech(batch):

    # load and resample audio data from 48 to 16kHz
    audio_1 = batch["audio"]
    audio_2 = data["train"][random.randint(0, len(data["train"])-1)]["audio"]

    # compute log-Mel input features from input audio array 
    spec_1 = feature_extractor(audio_1["array"], sampling_rate=audio_1["sampling_rate"]).input_features[0]
    spec_2 = feature_extractor(audio_2["array"], sampling_rate=audio_2["sampling_rate"]).input_features[0]

    # plot_spect(MixSpeech(spec_1,spec_2), title="Mel Spectrogram", ylabel="Mel Filter", aspect="auto", xmax=None)

    batch["input_features"] = MixSpeech(spec_1,spec_2)

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch

def prepare_dataset_SpecAugment_MixSpeech(batch):

    # load and resample audio data from 48 to 16kHz
    audio_1 = batch["audio"]
    audio_2 = data["train"][random.randint(0, len(data["train"])-1)]["audio"]

    # compute log-Mel input features from input audio array 
    spec_1 = feature_extractor(audio_1["array"], sampling_rate=audio_1["sampling_rate"]).input_features[0]
    spec_2 = feature_extractor(audio_2["array"], sampling_rate=audio_2["sampling_rate"]).input_features[0]
    
    freq_masking_min_percentage = 0.002
    freq_masking_max_percentage = 0.010
    num_freq_mask = 16

    freq_percentage = random.uniform(freq_masking_min_percentage, freq_masking_max_percentage)
    # batch["input_features"] = freq_spec_augment(batch["input_features"], freq_percentage = freq_percentage)
    # plot_spectrogram(batch["input_features"], title="Mel Spectrogram", ylabel="Mel Filter", aspect="auto", xmax=None)

    time_masking_min_percentage = 0.02
    time_masking_max_percentage = 0.03
    num_time_mask = 4

    time_percentage = random.uniform(time_masking_min_percentage, time_masking_max_percentage)

    batch["input_features"] = MixSpeech(spec_1,spec_2)

    batch["input_features"] = SpecAugment(batch["input_features"], num_freq_mask, freq_percentage, num_time_mask, time_percentage)

    # plot_spect(batch["input_features"], title="Log-Mel Spectrogram", ylabel="Mel Filter", aspect="auto", xmax=None)
        
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch

def prepare_dataset_SpecSwap(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_features"] = SpecSwap(batch["input_features"], 0.3, 0.3)
    # plot_spect(batch["input_features"], title="Mel Spectrogram", ylabel="Mel Filter", aspect="auto", xmax=None)

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    return batch

if cfg.get("augmentations") in ["SpecAugment"]:
    print("using SpecAugment...")
    mapped_train = data['train'].map(prepare_dataset_SpecAugment, remove_columns=data.column_names["train"], num_proc=1)

    print("done Augmenting")
    mapped_test = data['validate'].map(prepare_dataset, num_proc=1)
    data = {'train': mapped_train, 'validate': mapped_test}

elif cfg.get("augmentations") in ["MixSpeech"]:
    print("using MixSpeech...")
    mapped_train = data['train'].map(prepare_dataset_MixSpeech, remove_columns=data.column_names["train"], num_proc=1)

    print("done Augmenting")
    mapped_test = data['validate'].map(prepare_dataset, num_proc=1)
    data = {'train': mapped_train, 'validate': mapped_test}

elif cfg.get("augmentations") in ["SpecAugment_MixSpeech"]:

    print("using SpecAugment_MixSpeech...")
    mapped_train = data['train'].map(prepare_dataset_SpecAugment_MixSpeech, remove_columns=data.column_names["train"], num_proc=1)

    print("done Augmenting")
    mapped_test = data['validate'].map(prepare_dataset, num_proc=1)
    data = {'train': mapped_train, 'validate': mapped_test}

elif cfg.get("augmentations") in ["SpecSwap"]:
    print("using SpecSwap...")
    mapped_train = data['train'].map(prepare_dataset_SpecSwap, remove_columns=data.column_names["train"], num_proc=1)

    print("done Augmenting")
    mapped_test = data['validate'].map(prepare_dataset, num_proc=1)
    data = {'train': mapped_train, 'validate': mapped_test}

else:
    print("base mapping...")
    data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=1)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        del input_features
        del labels_batch
        del labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load(os.path.join(script_path,"wer.py"), keep_in_memory=False)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = WhisperForConditionalGeneration.from_pretrained(model_path, use_cache=False)
config = WhisperConfig.from_pretrained(model_path)

if cfg.get("with_drop_out"): 

    config.dropout = cfg.get("drop_out") 
    model.config = config

# print the model dropout config
print("the current dropout Value is: ".format(model.config.dropout))

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

##############################################################################################################

training_args = Seq2SeqTrainingArguments(
    output_dir=out_put_model,
    per_device_train_batch_size= cfg.get("batch_size"),
    gradient_accumulation_steps= grad_cal(cfg.get("batch_size")),
    learning_rate=float(cfg.get("learning_rate")),
    weight_decay= cfg.get("weight_decay"),
    max_grad_norm= cfg.get("max_grad_norm"),
    warmup_steps=cfg.get("warmup_steps"), 
    max_steps= num_of_max_steps,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_eval_batch_size=cfg.get("batch_size"),
    predict_with_generate=True,
    generation_max_length=225,
    save_steps= num_val_save_steps,
    eval_steps= num_val_save_steps,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=cfg.get("save_total_limit"), 
    optim= cfg.get("optim") 
)


# Define a custom callback to stop training after a certain number of steps
class StopTrainingCallback(TrainerControl):
    def __init__(self, num_worse_epochs=2):
        # self.num_steps = num_steps
        self.save_count = 0
        self.num_worse_epochs = num_worse_epochs
        self.worse_epoch_count = 0
        self.best_validation_loss = float("inf")

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        pass
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        pass
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        validation_loss = state.log_history[-1]["eval_loss"]

        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.worse_epoch_count = 0
        else:
            self.worse_epoch_count += 1

        if self.worse_epoch_count >= self.num_worse_epochs:
            control.should_training_stop = True
    def on_init_end(self, args, state, control, model=None, **kwargs):
        pass
    def on_log(self, args, state, control, model=None, **kwargs):
        pass
    def on_predict(self, args, state, control, model=None, **kwargs):
        pass
    def on_prediction_step(self, args, state, control, model=None, **kwargs):
        pass
    def on_save(self, args, state, control, model=None, **kwargs):
        # self.save_count += 1
        # save_limit = int(5 * epoch_per_cycle)
        # if self.save_count >= save_limit:
        #     control.should_save=True
        #     control.should_training_stop = True
        pass
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        pass
    def on_step_end(self, args, state, control, model=None, **kwargs):
        pass
    def on_substep_end(self, args, state, control, model=None, **kwargs):
        pass
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        pass
    def on_train_end(self, args, state, control, model=None, **kwargs):
        pass

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["validate"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    # callbacks=[StopTrainingCallback],
)

processor.save_pretrained(training_args.output_dir)
model.config.save_pretrained(training_args.output_dir)

if cfg.get("from_checkpoint"):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

trainer.save_model(training_args.output_dir)