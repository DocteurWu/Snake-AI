# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Démonstration de l'IA
=============================================================================
 Charge un modèle entraîné (DQN ou Imitation) et le fait jouer en mode
 graphique pour visualiser ses performances.
 
 Usage :
   python demo_ia.py                  -> Charger le modèle DQN
   python demo_ia.py --imitation      -> Charger le modèle Imitation
   python demo_ia.py --parties 10     -> Jouer 10 parties
   python demo_ia.py --vitesse 10     -> Vitesse réduite (10 FPS)
=============================================================================
"""

import argparse
import time
import sys

import torch
import numpy as np

from game import SnakeGame
from model import ReseauSerpent
from config import (
    DEVICE, NB_ACTIONS,
    NOM_MODELE_DQN, NOM_MODELE_IMITATION,
    VITESSE_IA
)


def demo_ia(nom_modele, nb_parties=5, vitesse=10):
    """
    Fait jouer l'IA en mode graphique.
    
    Paramètres :
        nom_modele (str) : Nom du fichier modèle à charger
        nb_parties (int)  : Nombre de parties à jouer
        vitesse (int)     : FPS (vitesse d'affichage)
    """
    print("=" * 60)
    print("SNAKE IA -- Mode Demonstration")
    print(f"   Modele   : {nom_modele}")
    print(f"   Parties  : {nb_parties}")
    print(f"   Vitesse  : {vitesse} FPS")
    print("=" * 60)
    
    # --- Charger le modèle ---
    modele = ReseauSerpent()
    modele.charger(nom_modele)
    modele.eval()
    
    # --- Initialiser le jeu ---
    jeu = SnakeGame(mode_graphique=True, vitesse=vitesse)
    
    scores = []
    
    try:
        for partie in range(1, nb_parties + 1):
            etat = jeu.reset()
            
            while True:
                # --- L'IA choisit la meilleure action (pas d'exploration) ---
                with torch.no_grad():
                    etat_tensor = torch.tensor(
                        etat, dtype=torch.float32, device=DEVICE
                    ).unsqueeze(0)
                    q_values = modele(etat_tensor)
                    action = q_values.argmax(dim=1).item()
                
                # --- Exécuter l'action ---
                etat, recompense, termine, score = jeu.step(action)
                
                # --- Affichage ---
                continuer = jeu.render()
                if not continuer:
                    jeu.close()
                    return
                
                if termine:
                    scores.append(score)
                    print(f"   Partie {partie}/{nb_parties} -- Score: {score}")
                    time.sleep(1)  # Pause entre les parties
                    break
    
    except KeyboardInterrupt:
        print("\n[!] Demonstration interrompue.")
    
    finally:
        jeu.close()
        
        if scores:
            print("\n" + "=" * 60)
            print("RESULTATS")
            print(f"   Parties jouees  : {len(scores)}")
            print(f"   Score moyen     : {sum(scores)/len(scores):.1f}")
            print(f"   Meilleur score  : {max(scores)}")
            print(f"   Pire score      : {min(scores)}")
            print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration de l'IA Snake")
    parser.add_argument(
        "--imitation", action="store_true",
        help="Utiliser le modele d'imitation au lieu du DQN"
    )
    parser.add_argument(
        "--parties", type=int, default=5,
        help="Nombre de parties a jouer (defaut: 5)"
    )
    parser.add_argument(
        "--vitesse", type=int, default=10,
        help="Vitesse d'affichage en FPS (defaut: 10)"
    )
    
    args = parser.parse_args()
    
    nom = NOM_MODELE_IMITATION if args.imitation else NOM_MODELE_DQN
    demo_ia(nom, nb_parties=args.parties, vitesse=args.vitesse)
