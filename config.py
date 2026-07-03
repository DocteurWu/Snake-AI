# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Fichier de Configuration
=============================================================================
 Centralise tous les hyperparamètres et constantes du projet.
 Optimisé pour Orange Pi 3B (ARM, 4 cœurs, CPU uniquement).
=============================================================================
"""

import torch

# ========================== GRILLE DE JEU ==========================
TAILLE_GRILLE = 20          # Nombre de cases par côté (grille carrée)
TAILLE_CASE = 20            # Taille d'une case en pixels (mode graphique)
VITESSE_JEU = 18            # FPS en mode graphique (jeu humain rapide)
VITESSE_IA = 0              # FPS en mode IA graphique (0 = max)
MAX_STEPS_SANS_MANGER = 100 # Nombre de steps max sans manger avant game over
                             # (évite les boucles infinies de l'IA)

# ========================== COULEURS (Pygame) ==========================
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)
ROUGE = (200, 50, 50)
VERT_FONCE = (0, 150, 0)
VERT_CLAIR = (0, 200, 0)
GRIS = (40, 40, 40)
BLEU = (50, 100, 200)

# ========================== DIRECTIONS (absolues internes) ==========================
# Utilisées en interne uniquement. Le serpent agit en RELATIF.
DROITE = (1, 0)
GAUCHE = (-1, 0)
HAUT = (0, -1)
BAS = (0, 1)

# Mapping circulaire pour les rotations (sens horaire)
SENS_HORAIRE = [DROITE, BAS, GAUCHE, HAUT]

# ========================== ACTIONS RELATIVES ==========================
TOUT_DROIT = 0
TOURNER_GAUCHE = 1
TOURNER_DROITE = 2
NB_ACTIONS = 3

# ========================== ÉTAT (STATE) ==========================
# Taille du vecteur d'état :
# 3 (dangers) + 4 (direction one-hot) + 4 (position nourriture relative)
TAILLE_ETAT = 11

# ========================== RÉCOMPENSES ==========================
RECOMPENSE_MANGER = 10.0
RECOMPENSE_MOURIR = -10.0
RECOMPENSE_STEP = 0.0       # Récompense neutre par step (pas de pénalité)
RECOMPENSE_RAPPROCHEMENT = 1.0   # Bonus si on se rapproche de la nourriture
RECOMPENSE_ELOIGNEMENT = -1.0    # Malus si on s'éloigne

# ========================== MODÈLE NEURONAL ==========================
COUCHES_CACHEES = [256, 128]  # Architecture du réseau (léger pour ARM)
TAUX_APPRENTISSAGE = 0.001

# ========================== DQN — Hyperparamètres ==========================
TAILLE_MEMOIRE = 100_000     # Capacité du replay buffer
TAILLE_BATCH = 64            # Taille du mini-batch
GAMMA = 0.99                 # Facteur de discount
EPSILON_DEBUT = 1.0          # Exploration initiale
EPSILON_FIN = 0.01           # Exploration minimale
EPSILON_DECROISSANCE = 0.995 # Facteur de décroissance par épisode
MISE_A_JOUR_CIBLE = 10       # Fréquence de sync du réseau cible (en épisodes)

# ========================== IMITATION LEARNING ==========================
CHEMIN_DONNEES_IMITATION = "data/parties_humaines.pkl"  # Fichier de sauvegarde
EPOCHS_IMITATION = 50        # Nombre d'époques d'entraînement
TAILLE_BATCH_IMITATION = 32

# ========================== SAUVEGARDES ==========================
CHEMIN_MODELE = "models/"
NOM_MODELE_DQN = "dqn_snake.pth"
NOM_MODELE_IMITATION = "imitation_snake.pth"

# ========================== DEVICE (CPU uniquement pour Orange Pi) ==========================
DEVICE = torch.device("cpu")  # Pas de GPU sur Orange Pi

# ========================== MULTITHREADING ==========================
# Limite PyTorch à 4 threads (= 4 cœurs du Orange Pi 3B)
THREADS_CPU = 4
torch.set_num_threads(THREADS_CPU)
torch.set_num_interop_threads(THREADS_CPU)
