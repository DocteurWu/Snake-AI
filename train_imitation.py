# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement par Imitation (Behavioral Cloning)
=============================================================================
 Entraîne le réseau de neurones à reproduire les actions du joueur humain
 à partir des données enregistrées par play_and_record.py.
 
 Usage :
   python train_imitation.py
   python train_imitation.py --epochs 100
=============================================================================
"""

import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import ReseauSerpent
from config import (
    DEVICE, NB_ACTIONS, TAILLE_ETAT,
    TAUX_APPRENTISSAGE, CHEMIN_DONNEES_IMITATION,
    EPOCHS_IMITATION, TAILLE_BATCH_IMITATION,
    NOM_MODELE_IMITATION
)


class DatasetImitation(Dataset):
    """
    Dataset PyTorch pour les données d'imitation.
    Charge les paires (état, action) enregistrées par le joueur humain.
    """
    
    def __init__(self, chemin_donnees):
        """
        Charge les données depuis un fichier pickle.
        
        Paramètres :
            chemin_donnees (str) : Chemin vers le fichier .pkl
        """
        if not os.path.exists(chemin_donnees):
            raise FileNotFoundError(
                f"Fichier de donnees introuvable : {chemin_donnees}\n"
                f"Lancez d'abord 'python play_and_record.py' pour enregistrer des parties."
            )
        
        with open(chemin_donnees, "rb") as f:
            donnees_brutes = pickle.load(f)
        
        print(f"[OK] {len(donnees_brutes)} transitions chargees depuis {chemin_donnees}")
        
        # Séparer les états et les actions
        self.etats = np.array([d[0] for d in donnees_brutes], dtype=np.float32)
        self.actions = np.array([d[1] for d in donnees_brutes], dtype=np.int64)
        
        # Afficher la distribution des actions
        noms_actions = ["Tout droit", "Tourner gauche", "Tourner droite"]
        print("\n   Distribution des actions :")
        for i in range(NB_ACTIONS):
            count = np.sum(self.actions == i)
            pct = 100 * count / len(self.actions)
            print(f"     {noms_actions[i]:>16} : {count:>6} ({pct:.1f}%)")
    
    def __len__(self):
        return len(self.etats)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.etats[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.long)
        )


def entrainer_imitation(nb_epochs=EPOCHS_IMITATION):
    """
    Entraîne le modèle par Behavioral Cloning (classification supervisée).
    
    L'idée : le réseau apprend à prédire l'action du joueur humain
    à partir du vecteur d'état. C'est un simple problème de classification
    à 3 classes (tout_droit, gauche, droite).
    
    Paramètres :
        nb_epochs (int) : Nombre d'époques d'entraînement
    """
    print("=" * 60)
    print("SNAKE IA -- Entrainement par Imitation")
    print("=" * 60)
    
    # --- Charger les données ---
    dataset = DatasetImitation(CHEMIN_DONNEES_IMITATION)
    
    # --- Séparer en train/validation (80/20) ---
    taille_total = len(dataset)
    taille_train = int(0.8 * taille_total)
    taille_val = taille_total - taille_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [taille_train, taille_val]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TAILLE_BATCH_IMITATION,
        shuffle=True,
        num_workers=0,   # Pas de multiprocessing (ARM léger)
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TAILLE_BATCH_IMITATION,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\n   Donnees train      : {taille_train}")
    print(f"   Donnees validation : {taille_val}")
    
    # --- Modèle, optimiseur, perte ---
    modele = ReseauSerpent()
    optimiseur = optim.Adam(modele.parameters(), lr=TAUX_APPRENTISSAGE)
    critere = nn.CrossEntropyLoss()  # Classification à 3 classes
    
    nb_params = sum(p.numel() for p in modele.parameters())
    print(f"\n   Modele : {nb_params:,} parametres")
    print(f"   Epoques : {nb_epochs}")
    print(f"   Batch size : {TAILLE_BATCH_IMITATION}")
    print()
    
    meilleure_precision = 0.0
    
    for epoch in range(1, nb_epochs + 1):
        # === PHASE D'ENTRAÎNEMENT ===
        modele.train()
        perte_totale = 0.0
        correct = 0
        total = 0
        
        for etats_batch, actions_batch in train_loader:
            etats_batch = etats_batch.to(DEVICE)
            actions_batch = actions_batch.to(DEVICE)
            
            # Forward
            predictions = modele(etats_batch)
            perte = critere(predictions, actions_batch)
            
            # Backward
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()
            
            perte_totale += perte.item() * etats_batch.size(0)
            correct += (predictions.argmax(dim=1) == actions_batch).sum().item()
            total += etats_batch.size(0)
        
        perte_train = perte_totale / total
        precision_train = 100 * correct / total
        
        # === PHASE DE VALIDATION ===
        modele.eval()
        perte_val_totale = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for etats_batch, actions_batch in val_loader:
                etats_batch = etats_batch.to(DEVICE)
                actions_batch = actions_batch.to(DEVICE)
                
                predictions = modele(etats_batch)
                perte = critere(predictions, actions_batch)
                
                perte_val_totale += perte.item() * etats_batch.size(0)
                correct_val += (predictions.argmax(dim=1) == actions_batch).sum().item()
                total_val += etats_batch.size(0)
        
        perte_val = perte_val_totale / total_val if total_val > 0 else 0
        precision_val = 100 * correct_val / total_val if total_val > 0 else 0
        
        # --- Affichage ---
        print(
            f"Epoque {epoch:>3}/{nb_epochs} | "
            f"Train: perte={perte_train:.4f}, acc={precision_train:.1f}% | "
            f"Val: perte={perte_val:.4f}, acc={precision_val:.1f}%"
        )
        
        # --- Sauvegarder le meilleur modèle ---
        if precision_val > meilleure_precision:
            meilleure_precision = precision_val
            modele.sauvegarder(NOM_MODELE_IMITATION)
            print(f"   -> Meilleur modele sauvegarde (val acc: {precision_val:.1f}%)")
    
    print("\n" + "=" * 60)
    print("RESUME")
    print(f"   Meilleure precision validation : {meilleure_precision:.1f}%")
    print(f"   Modele sauvegarde : models/{NOM_MODELE_IMITATION}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrainement par imitation")
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS_IMITATION,
        help=f"Nombre d'epoques (defaut: {EPOCHS_IMITATION})"
    )
    args = parser.parse_args()
    
    entrainer_imitation(nb_epochs=args.epochs)
