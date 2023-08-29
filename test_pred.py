import torch
from typing import List
from transformers import BertTokenizer
import regex as re
import args
from tokenizer.tokenizer import MolTranBertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
import pandas as pd
import numpy as np
from finetune_pubchem_light import LightningModule, PropertyPredictionDataModule

def load_pretrained_model(checkpoint_path, model_architecture):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_architecture.to(device)
    print("Loading model checkpoint from:", checkpoint_path)
    print("Using device:", device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("checkpoint: ", checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def predict_smiles(datamodule, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with torch.no_grad():
        predictions = []
        y_test = []
        for idx, mask, labels in datamodule.test_dataloader():
            idx = idx.to(device)
            mask = mask.to(device)

            output = model(idx, mask)
            prediction = output.squeeze().tolist()
            predictions.extend(prediction)
            y_test.extend(labels.tolist())
            
        preds = [1 if pred >= 0.5 else 0 for pred in predictions]
        
        probs = predictions  
        evaluation_metrics = {
            'Acc': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'AUC': roc_auc_score(y_test, probs),
            'F1-score': f1_score(y_test, preds),
            'AP': average_precision_score(y_test, probs)
        }
        
        print("Evaluation Metrics:")
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value}")

    return {"predictions": predictions}

def main():
    
    
    margs = args.parse_args()
    
    datamodule = PropertyPredictionDataModule(margs)
    
    #for batch in datamodule.test_dataloader():
        #print("batch: ", batch)
        #break
    
    config = margs
    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    model_architecture = LightningModule(config, tokenizer)
    checkpoint_path = 'last.ckpt'

    model = load_pretrained_model(checkpoint_path, model_architecture)
    
    predictions = predict_smiles(datamodule, model)

if __name__ == "__main__":
    main()
    