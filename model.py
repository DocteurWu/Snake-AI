# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Modèle Neuronal
=============================================================================
 Réseau de neurones léger optimisé pour CPU ARM (Orange Pi 3B).
 Utilisé à la fois pour le DQN et l'Apprentissage par Imitation.
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    TAILLE_ETAT, NB_ACTIONS, COUCHES_CACHEES,
    DEVICE, CHEMIN_MODELE
)


class ReseauSerpent(nn.Module):
    """
    Réseau de neurones feed-forward léger.
    
    Architecture :
        Entrée (11) -> Couche cachée 1 (256) -> Couche cachée 2 (128) -> Sortie (3)
    
    Activations : ReLU (rapide sur CPU)
    Pas de Dropout pour garder le modèle léger.
    """
    
    def __init__(self):
        super(ReseauSerpent, self).__init__()
        
        # --- Construction dynamique des couches ---
        tailles = [TAILLE_ETAT] + COUCHES_CACHEES + [NB_ACTIONS]
        couches = []
        
        for i in range(len(tailles) - 1):
            couches.append(nn.Linear(tailles[i], tailles[i + 1]))
            # Pas de ReLU après la dernière couche (sortie brute = Q-values)
            if i < len(tailles) - 2:
                couches.append(nn.ReLU())
        
        self.reseau = nn.Sequential(*couches)
        
        # --- Déplacer sur le device (CPU) ---
        self.to(DEVICE)
    
    def forward(self, x):
        """
        Propagation avant.
        
        Paramètres :
            x (Tensor) : Vecteur d'état [batch, 11]
        
        Retourne :
            Tensor : Q-values ou logits [batch, 3]
        """
        return self.reseau(x)
    
    def sauvegarder(self, nom_fichier):
        """
        Sauvegarde les poids du modèle.
        
        Paramètres :
            nom_fichier (str) : Nom du fichier (sera placé dans CHEMIN_MODELE)
        """
        os.makedirs(CHEMIN_MODELE, exist_ok=True)
        chemin = os.path.join(CHEMIN_MODELE, nom_fichier)
        torch.save(self.state_dict(), chemin)
        print(f"[OK] Modele sauvegarde : {chemin}")
    
    def charger(self, nom_fichier):
        """
        Charge les poids d'un modèle sauvegardé.
        
        Paramètres :
            nom_fichier (str) : Nom du fichier à charger
        """
        chemin = os.path.join(CHEMIN_MODELE, nom_fichier)
        if os.path.exists(chemin):
            self.load_state_dict(torch.load(chemin, map_location=DEVICE))
            self.eval()
            print(f"[OK] Modele charge : {chemin}")
        else:
            print(f"[!!] Fichier introuvable : {chemin}")


# =============================================================================
# Test rapide du modèle
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Test du modele ReseauSerpent")
    print("=" * 50)
    
    modele = ReseauSerpent()
    print(f"\nArchitecture :\n{modele}")
    
    # Compter les paramètres
    nb_params = sum(p.numel() for p in modele.parameters())
    print(f"\nNombre de parametres : {nb_params:,}")
    
    # Test avec un état fictif
    etat_test = torch.randn(1, TAILLE_ETAT)
    sortie = modele(etat_test)
    print(f"\nEntree  : {etat_test.shape}")
    print(f"Sortie  : {sortie.shape}")
    print(f"Q-values: {sortie.detach().numpy()}")
    
    # Test avec un batch
    batch_test = torch.randn(64, TAILLE_ETAT)
    sortie_batch = modele(batch_test)
    print(f"\nBatch entree  : {batch_test.shape}")
    print(f"Batch sortie  : {sortie_batch.shape}")
    
    print("\n[OK] Tous les tests passes !")
