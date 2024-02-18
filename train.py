
from transformerModel import Transformer
import helper

import torch
import torch.nn as nn
import torch.utils.data as data
import os
# from pathlib import Path
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

### comfiguration parameters
config = helper.AttrDict()
config.device = torch.device(device)
config.currentPath = os.path.dirname(os.path.abspath(__file__))
config.dataSetFileName = 'english_french_sentences.csv'
config.sourceLang = 'en'
config.targetLang = 'fr'
config.d_model = 512
config.num_heads = 8
config.num_layers = 6
config.trainBatchSize = 8
config.valBatchSize = 1
config.d_ff = 2048
config.max_seq_length = 100
config.dropout = 0.1
config.lr = 0.0001
config.num_epochs = 20
config.trainPercentage = 0.9 # 90% for training

# Data preparation
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = helper.get_data_ready(config)

model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config)

model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)

model.train()

for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch + 1:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(config.device)
            decoder_input = batch['decoder_input'].to(config.device)
            encoder_mask = batch['encoder_mask'].to(config.device)
            decoder_mask = batch['decoder_mask'].to(config.device)
            label = batch['label'].to(config.device)

            output, _, _ = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = criterion(output.contiguous().view(-1, tokenizer_tgt.get_vocab_size()), label.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Loss: {average_loss}")

        # Save the model at the end of every epoch
        torch.save(model.state_dict(), f'{config.currentPath}/model/TrainedTransformerModelTranslateing_{config.sourceLang}_to_{config.targetLang}_epoch{epoch + 1}.pt')

        # Run validation at the end of every epoch
        helper.validate_model(model, val_dataloader, tokenizer_src, tokenizer_tgt, config)

print("Training finished.")