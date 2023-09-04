
import os
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch
import os
from tqdm import tqdm
from torch.optim import AdamW
import yaml
import argparse
import sys

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

def Mask_model_bert():

    model_name = os.path.join(script_path, "bert-base-german-cased")
    if not os.path.exists(model_name):
        print("the masked model does not exist need bert-base-german-cased folder next to the script")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    txt_file_name = cfg.get("txt_file_name")

    path = os.path.join(script_path, "text", txt_file_name)
    fp = open(path, "r", encoding="utf-8" )
    text = fp.read().split("\n")

    processed_data_path = os.path.join(script_path, "custom_dataset", "processed_" + txt_file_name.rsplit(".",1)[0] + ".pt")
    if os.path.exists(processed_data_path):
        # Load the inputs from the file
        print("loading existing preprocessed data...")
        inputs = torch.load(processed_data_path)
        print("done loading existing preprocessed data")
        
    else:
        print("preprocessing...")
        inputs = tokenizer(text, return_tensors= "pt", max_length = 512, truncation = True, padding = "max_length")
        # Save the inputs to a file
        torch.save(inputs, processed_data_path)
        print("done preprocessing")

    inputs["labels"] = inputs.input_ids.detach().clone()

    rand = torch.rand(inputs.input_ids.shape)

    mask_arr = (rand < cfg.get("random_masking_precentage")) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(mask_arr.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    for i in range(mask_arr.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    class MeditationsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        def __len__(self):
            return len(self.encodings.input_ids)
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    dataset = MeditationsDataset(inputs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = cfg.get("LM_batch_size"), shuffle = True)
    optim = AdamW(model.parameters(), lr = float(cfg.get("LM_learning_rate")))

    epochs = cfg.get("LM_epoch")
    train_loss = []

    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, leave = True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs.loss
            loss.backward()
            
            optim.step()

            loop.set_description("epoch {}".format(epoch))
            loop.set_postfix(loss=f' {loss.item():.4f}')
            train_loss.append(loss.item())

        model.save_pretrained(os.path.join(model_name, txt_file_name.rsplit(".",1)[0], str(epoch)))
        tokenizer.save_pretrained(os.path.join(model_name, txt_file_name.rsplit(".",1)[0], str(epoch)))

        # Save the loss values to a file
        loss_file = open(os.path.join(model_name, txt_file_name.rsplit(".",1)[0], str(epoch), "loss_values.txt"), "w")
        for loss_value in train_loss:
            loss_file.write(str(f'{loss_value:.4f}') + "\n")
        train_loss = []

Mask_model_bert()


