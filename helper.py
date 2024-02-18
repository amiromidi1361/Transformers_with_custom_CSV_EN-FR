import os
from pathlib import Path
import pandas as pd
import torch
import torchmetrics

from sklearn.model_selection import train_test_split
import pyarrow as pa
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset as TDataset

'''
Most of the code used in this file are taken from: 
https://github.com/hkproj/pytorch-transformer
'''

class BilingualDataset(TDataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, config):
        super().__init__()
        self.seq_len = config.max_seq_length

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = config.sourceLang
        self.tgt_lang = config.targetLang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path("tokenizer_{0}.json".format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_data_ready(config):

    currentPath = os.path.dirname(os.path.abspath(__file__))
    print(currentPath)
    path_to_file = Path(f"{config.currentPath}/{config.dataSetFileName}")

    csv = pd.read_csv(path_to_file)
    ds_raw = Dataset(pa.Table.from_pandas(csv))
     
    tokenizer_src = get_or_build_tokenizer(ds_raw, config.sourceLang)
    tokenizer_tgt = get_or_build_tokenizer(ds_raw, config.targetLang)

    train_ds_size = int(config.trainPercentage * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config)
    val_ds   = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['en']).ids
        tgt_ids = tokenizer_tgt.encode(item['fr']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config.trainBatchSize, shuffle=True)
    val_dataloader   = DataLoader(val_ds, batch_size=config.valBatchSize, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def validate_model2222(model, validation_ds, tokenizer_src, tokenizer_tgt, config):    
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(config.device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(config.device) # (b, 1, 1, seq_len)
            
            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            ##########################################################
            # model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            sos_idx = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64).to(config.device)
            eos_idx = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64).to(config.device)
            pad_idx = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64).to(config.device)
            decoder_input = torch.cat(
            [
                sos_idx,                                
                torch.tensor([pad_idx] * (config.max_seq_length - 1), dtype=torch.int64).to(config.device),
            ],dim=0,).to(config.device)
            decoder_input = decoder_input.unsqueeze(0)
              
            for i in range(config.max_seq_length-1):                                
                # print(i)
                # build mask for target
                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(config.device)
                # print('decoder_mask Val: ',decoder_mask)
                # calculate output                
                output, _, _ = model(encoder_input, decoder_input,encoder_mask, decoder_mask)
                # print('output',output.shape)
                # get next token
                prob = output.contiguous().view(-1, tokenizer_tgt.get_vocab_size()).to(config.device)
                next_word = torch.argmax(prob[i+1,:])               
                print('next word:',next_word)
                decoder_input[0,i+1] = next_word
                # print('model_out: ',decoder_input.shape)

                if next_word == eos_idx:
                    break
            ##########################################################
            
            model_out  = decoder_input.squeeze(0)
            # print('model_out: ',model_out)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output            
            print(f"SOURCE: {source_text}")
            print(f"TARGET: {target_text}")
            print(f"PREDICTED: {model_out_text}")

            if count == 2:                
                break
    
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)


    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)


    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)

    print('\n\n\t\t-----------------------------')
    print('\t\tCharacter Error Rate is: ', cer)
    print('\t\tWord Error Rate is:      ', wer)
    print('\t\tBlue score is:           ', bleu)
    
def validate_model(model, validation_ds, tokenizer_src, tokenizer_tgt, config):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count = count + 1
            encoder_input = batch["encoder_input"].to(config.device)
            encoder_mask = batch["encoder_mask"].to(config.device)

            # SOS, EOS, and PAD tokens
            sos_idx = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64).to(config.device)
            eos_idx = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64).to(config.device)
            pad_idx = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64).to(config.device)

            # Initialize the decoder input with SOS and PAD tokens
            decoder_input = torch.cat(
                [
                    sos_idx,
                    torch.tensor([pad_idx] * (config.max_seq_length - 1), dtype=torch.int64).to(config.device),
                ],
                dim=0,
            ).unsqueeze(0).to(config.device)

            for i in range(config.max_seq_length - 1):
                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(config.device)
                output, _, _ = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                prob = output.contiguous().view(-1, tokenizer_tgt.get_vocab_size())
                next_word = torch.argmax(prob[i + 1, :])
                decoder_input[0, i + 1] = next_word

                if next_word == eos_idx:
                    break

            model_out = decoder_input.squeeze(0)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target, and model output
            print(f"SOURCE: {source_text}")
            print(f"TARGET: {target_text}")
            print(f"PREDICTED: {model_out_text}")

            if count == 2:                
                break
    
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)


    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)


    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)

    print('\n\n\t\t-----------------------------')
    print('\t\tCharacter Error Rate is: ', cer)
    print('\t\tWord Error Rate is:      ', wer)
    print('\t\tBlue score is:           ', bleu)