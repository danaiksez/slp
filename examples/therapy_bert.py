import numpy as np
import pandas as pd
import torch 
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose
from sklearn.model_selection import KFold
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

from slp.data.collators import SequenceClassificationCollator, BertCollator
from slp.data.therapy_bert import PsychologicalDataset, TupleDataset, preprocessing
from slp.data.transforms import ToTensor
from slp.util.embeddings import EmbeddingsLoader
from slp.trainer.trainer import BertTrainer

#DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COLLATE_FN = SequenceClassificationCollator(device=DEVICE)
COLLATE_FN1 = BertCollator(device=DEVICE)
DEBUG = False
KFOLD = True
MAX_EPOCHS = 50

def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN1)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN1)

    return train_loader, val_loader

def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.1, shuffle=True, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]

    return dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)


def kfold_split(dataset, batch_train, batch_val, k=5, shuffle=True, seed=None):
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_indices, val_indices in kfold.split(dataset):
        yield dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)

def trainer_factory(embeddings, device=DEVICE):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
#    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = BertAdam(model.parameters(), lr=0.001)

    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }

    trainer = BertTrainer(
        model,
        optimizer,
        checkpoint_dir='../checkpoints' if not DEBUG else None,
        metrics=metrics,
        non_blocking=True,
        patience=10,
        loss_fn=criterion,
        device=DEVICE)

    return trainer


if __name__ == '__main__':

    ####### Parameters ########
    batch_train = 8
    batch_val = 8

    max_sent_length = 500  #max number of sentences (turns) in transcript - after padding
    max_word_length = 150   #max length of each sentence (turn) - after padding
    num_classes = 2
    batch_size = 8
    hidden_size = 300

    epochs = 40

#    loader = EmbeddingsLoader('../data/glove.6B.300d.txt', 300)
    loader = EmbeddingsLoader('/data/embeddings/glove.840B.300d.txt', 300)
    word2idx, idx2word, embeddings = loader.load()
    embeddings = torch.tensor(embeddings)


    to_tensor = ToTensor(device=DEVICE)

    bio = PsychologicalDataset(
        '../data/balanced_new_csv.csv', '../data/psychotherapy/',
        text_transforms = to_tensor)


    lista = []
    for i, (t,label) in enumerate(bio):
#        import pdb; pdb.set_trace()
        j = 0
        while j <= len(t):
            lista.append((t[j:(j+509)], label))
            j += 509

    lista = preprocessing(lista, to_tensor)


    if KFOLD:
        cv_scores = []
        import gc
#        for train_loader, val_loader in kfold_split(bio, batch_train, batch_val):
        for train_loader, val_loader in kfold_split(lista, batch_train, batch_val):
            trainer = trainer_factory(embeddings, device=DEVICE)
            fold_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
            cv_scores.append(fold_score)
            print("**********************")
            print("edw")
            print(fold_score)
            del trainer
            gc.collect()
        final_score = float(sum(cv_scores)) / len(cv_scores)
    else:
        train_loader, val_loader = train_test_split(lista, batch_train, batch_val)
        trainer = trainer_factory(embeddings, device=DEVICE)
        final_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)

    print(f'Final score: {final_score}')




    if DEBUG:
        print("Starting end to end test")
        print("-----------------------------------------------------------------------")
        trainer.fit_debug(train_loader, val_loader)
        print("Overfitting single batch")
        print("-----------------------------------------------------------------------")
        trainer.overfit_single_batch(train_loader)
