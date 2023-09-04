import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, Seq2SeqTrainingArguments, pipeline
import torch
from datasets import load_from_disk
from transformers import Seq2SeqTrainer
import evaluate
from typing import Any, Dict, List, Union
from dataclasses import dataclass
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


model_cycle = cfg.get("test_model_name")

model_path = os.path.join(script_path, model_cycle)

dic_path = os.path.join(script_path, "custom_dataset/test_samples/dict")
out_put_model = os.path.join(script_path, "evaluation", model_cycle + "_tested")

dataset = load_from_disk(dic_path)

model = WhisperForConditionalGeneration.from_pretrained(model_path)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

processor = WhisperProcessor.from_pretrained(model_path, language="German", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(model_path, language="German", task="transcribe")

def prepare_dataset(batch):

    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, num_proc=1)

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

metric = evaluate.load(os.path.join(script_path,"wer.py"))

def compute_metrics(pred) -> Dict[str, Any]:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric._compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir=out_put_model,  
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps= 1250,
    eval_steps= 1250,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=2,
    optim= "adamw_torch"
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

trainer.evaluate(eval_dataset=dataset["test"])




































# tr = 0
# for sample in dataset["test"]:
#     audio = sample["audio"]
#     if tr == 1:
#         break
#     tr += 1
#     inputs = processor(audio["array"], sampling_rate= audio["sampling_rate"], return_tensors="pt")
#     inputs = inputs.to(device)

#     # Generate the transcript using the model
#     generated_ids = model.generate(inputs=inputs['input_features'], max_new_tokens=225)

#     predicted_transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     # Calculate WER
#     error = wer(sample["sentence"], predicted_transcript)

#     file.write("file: {}, transcript: {}\n".format(sample["path"].rsplit("/",1)[1], sample["sentence"]))
#     space = (len(sample["path"].rsplit("/",1)[1])-3) * " "
#     file.write("{} predicted_transcript: {}, WER: {}\n".format(space, predicted_transcript, error))

#     total_wer += error




























# sample = []

# with open(dic_path, 'rb') as d:
#     result = chardet.detect(d.read(1000))

# # Open the TSV file in read mode
# with open(dic_path, encoding= result['encoding'],  newline='') as f:
#     reader = csv.reader(f, delimiter='\t')

#     first_row = next(reader)

    
#     for i in range(1, 20000):
#         next(reader)
#     con = 0
#     for row in tqdm(reader):
#         if con == 20000:
#             break
#         sample_path = script_path + dataset_path + "clips/" + row[1]
#         sample_transcript = row[2]
#         sample.append((sample_path,sample_transcript))

#         con +=1


# def eval(dataset):

#     total_wer = 0

#     # path to the saved model directory
#     model_path = script_path + "/edits/whisper_DE_Audio-based_Augment_IIII"

#     # Load the fine-tuned model
#     model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Instantiate the processor
#     processor = WhisperProcessor.from_pretrained(model_path, language="German", task="transcribe")

#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     # Open the file in append mode (create a new file if it doesn't exist)
#     file_path = "{}_{}.txt".format(model_path, timestamp)

#     file = open(file_path, 'a', encoding='utf-8')

#     for audio_file, reference_transcript in tqdm(dataset, total=len(dataset)):

#         # Load and process the audio file
#         audio, sr = librosa.load(audio_file, sr=16000)

#         inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
#         inputs = inputs.to(device)
#         # Generate the transcript using the model
#         generated_ids = model.generate(inputs=inputs['input_features'], max_new_tokens=225)

#         predicted_transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#         # Calculate WER
#         error = wer(reference_transcript, predicted_transcript)

#         file.write("file: {}, transcript: {}\n".format(audio_file.rsplit("/",1)[1], reference_transcript))
#         space = (len(audio_file.rsplit("/",1)[1])-3) * " "
#         file.write("{} predicted_transcript: {}, WER: {}\n".format(space, predicted_transcript, error))

#         total_wer += error


#     # Calculate average WER
#     wer_score = total_wer / len(dataset)
#     print("Average WER:", round(wer_score))
#     file.write("Average WER: {}".format(round(wer_score)))

#     # Close the file
#     file.close()


# def recheck():
#     coun = 0
#     wer_sum = 0
#     file_path = script_path + "/edits/whisper_DE_base_IIII_2023-05-19_11-55-11.txt"

#     file = open(file_path, 'r', encoding='utf-8')
#     for line in file:
#         wer_result = (line.strip().rsplit(",",1)[1])
#         # print(wer_result[0:4])
#         if wer_result[1:4] == "WER":
#             coun +=1
#             wer_sum += float(wer_result[6:])
#     print("total number of transcripted lines: {} and avrage WER is {}".format(coun, round(wer_sum/coun,2)))

# eval(sample)
# recheck()