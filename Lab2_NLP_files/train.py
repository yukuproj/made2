import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.data import BucketIterator

import math
import time
import tqdm

import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
from IPython.display import clear_output


def train(
    model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None, transformer=False
):
    model.train()
    epoch_loss = 0
    history = []
    for i, batch in tqdm.tqdm(enumerate(iterator)):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        if transformer:
            output, _ = model(src, trg[:,:-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
        else:
            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
        history.append(loss.cpu().data.numpy())
        if (i+1)%20==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            plt.show()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, transformer=False):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            if transformer:
                output, _ = model(src, trg[:,:-1])
                output = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:,1:].contiguous().view(-1)
            else:
                output = model(src, trg, 0)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _len_sort_key(x):
        return len(x.src)
    

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)

        
def init_model_architecture(network, SRC, TRG, device, bert=False, bert_model=None):
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 768 if bert else 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    if bert:
        enc = network.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, bert_model)
    else:
        enc = network.Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = network.Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    # dont forget to put the model to the right device
    model = network.Seq2Seq(enc, dec, device).to(device)
    if bert:
        model.encoder.rnn.apply(init_weights)
        model.decoder.apply(init_weights)
    else:
        model.apply(init_weights)
    return model

    
def train_model(
    train_data, valid_data, test_data, SRC, TRG, network,
    model_name, batch_size=128, n_epochs=10, bert=False, bert_model=None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=batch_size, device=device, sort_key=_len_sort_key,
    )
    
    if bert:
        bert_model.pooler = nn.Linear(
            bert_model.pooler.dense.in_features, 768, bias=True
        )
        bert_model.encoder.layer = bert_model.encoder.layer[:3]
    
    model = init_model_architecture(network, SRC, TRG, device, bert, bert_model)

    PAD_IDX = TRG.vocab.stoi["<pad>"]
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, 
                              patience=2, min_lr=1e-6, verbose=True)

    
    train_history = []
    valid_history = []
    CLIP = 1

    best_valid_loss = float('inf')
    for epoch in range(n_epochs):    
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, train_history, valid_history)
        valid_loss = evaluate(model, valid_iterator, criterion)
        scheduler.step(valid_loss)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'artifacts/{model_name}.pt')

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    return model, test_iterator


def init_weights_xavier(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

        
def init_transformer_architecture(network, SRC, TRG, device):
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 4
    DEC_HEADS = 4
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = network.Encoder(INPUT_DIM, 
                  HID_DIM, 
                  ENC_LAYERS, 
                  ENC_HEADS, 
                  ENC_PF_DIM, 
                  ENC_DROPOUT, 
                  device)

    dec = network.Decoder(OUTPUT_DIM, 
                  HID_DIM, 
                  DEC_LAYERS, 
                  DEC_HEADS, 
                  DEC_PF_DIM, 
                  DEC_DROPOUT, 
                  device)
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = network.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.apply(init_weights_xavier)
    return model


def train_transformer(
    train_data, valid_data, test_data, SRC, TRG, network,
    model_name, batch_size=128, n_epochs=10
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size, device = device, sort_key=_len_sort_key)
    
    
    model = init_transformer_architecture(network, SRC, TRG,device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG.vocab.stoi[TRG.pad_token])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, 
                          patience=2, min_lr=1e-6, verbose=True)

    train_history = []
    valid_history = []
    CLIP = 1

    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP,
                           train_history, valid_history, transformer=True)
        valid_loss = evaluate(model, valid_iterator, criterion, transformer=True)
        scheduler.step(valid_loss)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'artifacts/{model_name}.pt')

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    return model, test_iterator


def load_best_model(experiment_name, network, SRC, TRG, test_iterator,
                    transformer=False, bert=False, bert_model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if transformer:
        best_model = init_transformer_architecture(network, SRC, TRG, device)
    elif bert:
        best_model = init_model_architecture(network, SRC, TRG, device, bert=True, bert_model=bert_model)
    else:
        best_model = init_model_architecture(network, SRC, TRG, device)

    best_model.load_state_dict(torch.load(f'artifacts/{experiment_name}.pt'))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi["<pad>"])
    test_loss = evaluate(best_model, test_iterator, criterion, transformer=transformer)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    return best_model
