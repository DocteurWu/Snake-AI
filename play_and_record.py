# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Mode Jeu Humain + Enregistrement Automatique (Vitesse +20%)
=============================================================================
 Permet de jouer au Snake et d'enregistrer automatiquement toutes les
 transitions (états, actions) pour l'apprentissage par imitation.
 
 Les données sont sauvegardées en continu et accumulées dans un fichier pickle.
 
 Contrôles :
   Flèches      : Diriger le serpent
   ESPACE       : Pause / Reprise du jeu
   ECHAP        : Quitter et sauvegarder
=============================================================================
"""

import os
import pickle
import sys
import pygame
import numpy as np

from game import SnakeGame
from config import (
    TAILLE_CASE, TAILLE_GRILLE, VITESSE_JEU,
    DROITE, GAUCHE, HAUT, BAS, SENS_HORAIRE,
    TOUT_DROIT, TOURNER_GAUCHE, TOURNER_DROITE,
    CHEMIN_DONNEES_IMITATION, BLANC
)

def convertir_touche_en_action_relative(touche, direction_actuelle_idx):
    if touche not in SENS_HORAIRE:
        return None
    idx_demande = SENS_HORAIRE.index(touche)
    if idx_demande == direction_actuelle_idx:
        return TOUT_DROIT
    if idx_demande == (direction_actuelle_idx + 1) % 4:
        return TOURNER_DROITE
    if idx_demande == (direction_actuelle_idx - 1) % 4:
        return TOURNER_GAUCHE
    return None

def jouer_et_enregistrer():
    print("=" * 60)
    print("SNAKE IA -- Enregistrement Direct & Automatique")
    print("   Vitesse   : 18 FPS (+20%)")
    print("   Contrôles : Flèches directionnelles")
    print("   ESPACE    : Pause")
    print("   ECHAP     : Quitter et sauvegarder")
    print("=" * 60)

    # Charger les données existantes
    donnees = []
    if os.path.exists(CHEMIN_DONNEES_IMITATION):
        try:
            with open(CHEMIN_DONNEES_IMITATION, "rb") as f:
                donnees = pickle.load(f)
            print(f"\n[✓] {len(donnees)} transitions globales déjà chargées.")
        except Exception:
            print("\n[i] Nouveau fichier de données créé.")

    jeu = SnakeGame(mode_graphique=True, vitesse=VITESSE_JEU)
    etat = jeu.reset()
    
    nb_parties = 0
    nb_transitions_session = 0
    en_cours = True
    en_pause = False
    
    mapping_touches = {
        pygame.K_RIGHT: DROITE,
        pygame.K_LEFT:  GAUCHE,
        pygame.K_UP:    HAUT,
        pygame.K_DOWN:  BAS
    }
    
    direction_souhaitee = None
    
    print("\n🐍 Jouez, chaque mouvement est automatiquement enregistré en direct !\n")
    
    while en_cours:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    en_cours = False
                elif event.key == pygame.K_SPACE:
                    en_pause = not en_pause
                    if en_pause:
                        print("   [PAUSE]")
                    else:
                        print("   [REPRISE]")
                elif event.key in mapping_touches:
                    direction_souhaitee = mapping_touches[event.key]
        
        if en_pause:
            jeu.horloge.tick(10)
            continue
            
        # Déterminer l'action relative
        if direction_souhaitee is not None:
            action = convertir_touche_en_action_relative(direction_souhaitee, jeu.index_direction)
            if action is None:
                action = TOUT_DROIT
        else:
            action = TOUT_DROIT
            
        # Enregistrer la transition (état courant + action prise)
        donnees.append((etat.copy(), action))
        nb_transitions_session += 1
        
        # Faire un pas dans le jeu
        etat_suivant, recompense, termine, score = jeu.step(action)
        jeu.render()
        
        direction_souhaitee = None
        etat = etat_suivant
        
        if termine:
            nb_parties += 1
            print(f"   💀 Partie {nb_parties} terminée — Score: {score} | Transitions: {nb_transitions_session}")
            etat = jeu.reset()

    # Écriture finale sur le disque
    if len(donnees) > 0:
        os.makedirs(os.path.dirname(CHEMIN_DONNEES_IMITATION), exist_ok=True)
        with open(CHEMIN_DONNEES_IMITATION, "wb") as f:
            pickle.dump(donnees, f)
        print(f"\n[FIN] Données sauvegardées dans {CHEMIN_DONNEES_IMITATION}")
        print(f"      Transitions session : {nb_transitions_session}")
        print(f"      Total en mémoire     : {len(donnees)}")
    else:
        print("\nAucune donnée enregistrée.")
        
    jeu.close()

if __name__ == "__main__":
    jouer_et_enregistrer()
