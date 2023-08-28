import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, average_precision_score
import pandas as pd
import numpy as np
from molformer.finetune.finetune_pubchem_light import LightningModule

def load_pretrained_model(checkpoint_path, model_architecture):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_architecture().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_smiles(smiles_list: List[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    with torch.no_grad():
        predictions = []
        for smiles in smiles_list:
            graph = smiles2graph(smiles)  
            x = torch.tensor(graph['node_feat'], dtype=torch.float).unsqueeze(0).to(device)
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).unsqueeze(0).to(device)
            edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.float).unsqueeze(0).to(device)

            output = model(x, edge_index, edge_attr)
            prediction = output.squeeze().item()
            predictions.append(prediction)
            
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
    
    model_architecture = LightningModule()
    checkpoint_path = 'last.ckpt'

    model = load_pretrained_model(checkpoint_path, model_architecture)
    
    df = pd.read_csv('DEL_v4.csv')

    smiles_list = df[df['Split'] == 'test']['SMILES'].tolist() 
    y_test = df[df['Split'] == 'test']['Activity'].to_numpy()
    
    predictions = predict_smiles(smiles_list, y_test)

if __name__ == "__main__":
    main()