# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Logique du Jeu
=============================================================================
 Classe SnakeGame : moteur du jeu Snake avec interface step(action).
 - Actions RELATIVES : [tout_droit, tourner_gauche, tourner_droite]
 - Capteurs : dangers, direction, position nourriture (relatifs)
 - Deux modes : graphique (Pygame) et headless (calcul pur)
=============================================================================
"""

import random
import numpy as np

from config import (
    TAILLE_GRILLE, TAILLE_CASE, VITESSE_JEU, VITESSE_IA,
    MAX_STEPS_SANS_MANGER,
    NOIR, BLANC, ROUGE, VERT_FONCE, VERT_CLAIR, GRIS, BLEU,
    DROITE, GAUCHE, HAUT, BAS, SENS_HORAIRE,
    TOUT_DROIT, TOURNER_GAUCHE, TOURNER_DROITE,
    RECOMPENSE_MANGER, RECOMPENSE_MOURIR, RECOMPENSE_STEP,
    RECOMPENSE_RAPPROCHEMENT, RECOMPENSE_ELOIGNEMENT,
    TAILLE_ETAT
)


class SnakeGame:
    """
    Moteur du jeu Snake.
    
    Interface principale :
        reset()           -> state : Réinitialise le jeu
        step(action)      -> (state, reward, done, score) : Avance d'un pas
        get_state()       -> np.array : Calcule le vecteur d'état actuel
        render()          -> None : Affiche le jeu (mode graphique uniquement)
        close()           -> None : Ferme proprement Pygame
    
    Paramètres :
        mode_graphique (bool) : True = affichage Pygame, False = headless
        vitesse (int)         : FPS (0 = vitesse maximale)
    """
    
    def __init__(self, mode_graphique=True, vitesse=None):
        """Initialise le jeu Snake."""
        self.mode_graphique = mode_graphique
        self.largeur = TAILLE_GRILLE  # en cases
        self.hauteur = TAILLE_GRILLE  # en cases
        
        # --- Initialisation Pygame (mode graphique uniquement) ---
        if self.mode_graphique:
            import pygame
            self.pygame = pygame
            pygame.init()
            self.fenetre = pygame.display.set_mode(
                (self.largeur * TAILLE_CASE, self.hauteur * TAILLE_CASE)
            )
            pygame.display.set_caption("Snake IA - Mode Graphique")
            self.horloge = pygame.time.Clock()
            self.police = pygame.font.SysFont('arial', 18)
            self.vitesse = vitesse if vitesse is not None else VITESSE_JEU
        else:
            # Mode headless : aucun import Pygame, calcul pur
            self.pygame = None
            self.fenetre = None
            self.horloge = None
            self.police = None
            self.vitesse = 0  # headless = pas de limite de FPS
        
        # --- Premier reset ---
        self.reset()
    
    def reset(self):
        """
        Réinitialise complètement le jeu.
        Retourne le vecteur d'état initial.
        """
        # Position initiale : centre de la grille
        centre_x = self.largeur // 2
        centre_y = self.hauteur // 2
        
        # Le serpent démarre avec 3 segments, dirigé vers la droite
        self.tete = (centre_x, centre_y)
        self.corps = [
            self.tete,
            (centre_x - 1, centre_y),
            (centre_x - 2, centre_y)
        ]
        
        # Direction initiale : DROITE (index 0 dans SENS_HORAIRE)
        self.index_direction = 0  # Index dans SENS_HORAIRE
        self.direction = SENS_HORAIRE[self.index_direction]
        
        # Nourriture
        self.nourriture = None
        self._placer_nourriture()
        
        # Compteurs
        self.score = 0
        self.steps = 0
        self.steps_sans_manger = 0
        self.game_over = False
        
        return self.get_state()
    
    def _placer_nourriture(self):
        """Place la nourriture aléatoirement sur une case libre."""
        while True:
            x = random.randint(0, self.largeur - 1)
            y = random.randint(0, self.hauteur - 1)
            if (x, y) not in self.corps:
                self.nourriture = (x, y)
                break
    
    def _est_collision(self, position):
        """
        Vérifie si une position donnée est en collision.
        Collision = mur ou corps du serpent.
        """
        x, y = position
        # Collision avec les murs
        if x < 0 or x >= self.largeur or y < 0 or y >= self.hauteur:
            return True
        # Collision avec le corps
        if position in self.corps:
            return True
        return False
    
    def _calculer_direction_relative(self, action):
        """
        Convertit une action relative en nouvelle direction absolue.
        
        Actions :
            0 (TOUT_DROIT)       : garde la direction actuelle
            1 (TOURNER_GAUCHE)   : tourne à gauche (sens anti-horaire)
            2 (TOURNER_DROITE)   : tourne à droite (sens horaire)
        """
        if action == TOUT_DROIT:
            # Pas de changement de direction
            pass
        elif action == TOURNER_GAUCHE:
            # Sens anti-horaire = index - 1 dans SENS_HORAIRE
            self.index_direction = (self.index_direction - 1) % 4
        elif action == TOURNER_DROITE:
            # Sens horaire = index + 1 dans SENS_HORAIRE
            self.index_direction = (self.index_direction + 1) % 4
        
        self.direction = SENS_HORAIRE[self.index_direction]
    
    def _position_relative(self, direction_relative):
        """
        Calcule la position adjacente dans une direction relative.
        
        direction_relative :
            'devant'  : dans la direction actuelle
            'gauche'  : à gauche de la direction actuelle
            'droite'  : à droite de la direction actuelle
        """
        if direction_relative == 'devant':
            idx = self.index_direction
        elif direction_relative == 'gauche':
            idx = (self.index_direction - 1) % 4
        elif direction_relative == 'droite':
            idx = (self.index_direction + 1) % 4
        else:
            idx = self.index_direction
        
        dx, dy = SENS_HORAIRE[idx]
        return (self.tete[0] + dx, self.tete[1] + dy)
    
    def get_state(self):
        """
        Calcule et retourne le vecteur d'état actuel.
        
        Vecteur de 11 valeurs :
        [0-2]  Dangers :    [devant, gauche, droite]           (bool -> 0/1)
        [3-6]  Direction :  [droite, bas, gauche, haut]        (one-hot)
        [7-10] Nourriture : [devant, derrière, gauche, droite] (bool -> 0/1)
        
        Toutes les directions sont RELATIVES à la tête du serpent.
        """
        # === 1. DANGERS (3 valeurs) ===
        # On vérifie la collision dans chaque direction relative
        danger_devant = self._est_collision(self._position_relative('devant'))
        danger_gauche = self._est_collision(self._position_relative('gauche'))
        danger_droite = self._est_collision(self._position_relative('droite'))
        
        # === 2. DIRECTION ACTUELLE — One-Hot Encoding (4 valeurs) ===
        dir_droite = self.direction == DROITE
        dir_bas    = self.direction == BAS
        dir_gauche = self.direction == GAUCHE
        dir_haut   = self.direction == HAUT
        
        # === 3. POSITION DE LA NOURRITURE — Relative (4 valeurs) ===
        # Vecteur nourriture - tête (en coordonnées absolues de la grille)
        dx_food = self.nourriture[0] - self.tete[0]
        dy_food = self.nourriture[1] - self.tete[1]
        
        # Direction actuelle du serpent (vecteur unitaire)
        dir_x, dir_y = self.direction
        
        # Directions perpendiculaires via les index (plus fiable que le calcul vectoriel)
        dir_gauche_vec = SENS_HORAIRE[(self.index_direction - 1) % 4]
        dir_droite_vec = SENS_HORAIRE[(self.index_direction + 1) % 4]
        
        # Projection du vecteur nourriture sur chaque direction relative
        # Nourriture devant ? (produit scalaire positif avec la direction actuelle)
        food_devant   = (dx_food * dir_x + dy_food * dir_y) > 0
        # Nourriture derrière ? (produit scalaire négatif)
        food_derriere = (dx_food * dir_x + dy_food * dir_y) < 0
        # Nourriture à gauche ? (produit scalaire positif avec la direction gauche)
        food_gauche   = (dx_food * dir_gauche_vec[0] + dy_food * dir_gauche_vec[1]) > 0
        # Nourriture à droite ? (produit scalaire positif avec la direction droite)
        food_droite   = (dx_food * dir_droite_vec[0] + dy_food * dir_droite_vec[1]) > 0
        
        # === Construction du vecteur d'état ===
        state = np.array([
            # Dangers (3)
            int(danger_devant),
            int(danger_gauche),
            int(danger_droite),
            # Direction actuelle — one-hot (4)
            int(dir_droite),
            int(dir_bas),
            int(dir_gauche),
            int(dir_haut),
            # Position nourriture relative (4)
            int(food_devant),
            int(food_derriere),
            int(food_gauche),
            int(food_droite)
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        Exécute une action et avance le jeu d'un pas.
        
        Paramètres :
            action (int) : 0=tout_droit, 1=tourner_gauche, 2=tourner_droite
        
        Retourne :
            state (np.array)  : Nouveau vecteur d'état (11 valeurs)
            reward (float)    : Récompense obtenue
            done (bool)       : True si la partie est terminée
            score (int)       : Score actuel
        """
        self.steps += 1
        self.steps_sans_manger += 1
        
        # --- Distance avant mouvement (pour reward shaping) ---
        dist_avant = abs(self.tete[0] - self.nourriture[0]) + \
                     abs(self.tete[1] - self.nourriture[1])
        
        # --- Appliquer l'action relative ---
        self._calculer_direction_relative(action)
        
        # --- Calculer la nouvelle position de la tête ---
        nouvelle_tete = (
            self.tete[0] + self.direction[0],
            self.tete[1] + self.direction[1]
        )
        
        # --- Vérifier la collision ---
        if self._est_collision(nouvelle_tete):
            self.game_over = True
            reward = RECOMPENSE_MOURIR
            state = self.get_state()
            return state, reward, True, self.score
        
        # --- Anti-boucle : trop de steps sans manger ---
        # Le seuil augmente avec la longueur du serpent
        if self.steps_sans_manger > MAX_STEPS_SANS_MANGER * len(self.corps):
            self.game_over = True
            reward = RECOMPENSE_MOURIR
            state = self.get_state()
            return state, reward, True, self.score
        
        # --- Déplacer le serpent ---
        self.tete = nouvelle_tete
        self.corps.insert(0, self.tete)
        
        # --- Vérifier si on mange la nourriture ---
        if self.tete == self.nourriture:
            self.score += 1
            self.steps_sans_manger = 0
            reward = RECOMPENSE_MANGER
            self._placer_nourriture()
        else:
            # Retirer la queue (le serpent avance sans grandir)
            self.corps.pop()
            
            # --- Reward shaping : rapprochement / éloignement ---
            dist_apres = abs(self.tete[0] - self.nourriture[0]) + \
                         abs(self.tete[1] - self.nourriture[1])
            if dist_apres < dist_avant:
                reward = RECOMPENSE_RAPPROCHEMENT
            elif dist_apres > dist_avant:
                reward = RECOMPENSE_ELOIGNEMENT
            else:
                reward = RECOMPENSE_STEP
        
        # --- Calculer le nouvel état ---
        state = self.get_state()
        
        return state, reward, False, self.score
    
    def render(self):
        """
        Affiche le jeu à l'écran (mode graphique uniquement).
        Gère aussi les événements Pygame (fermeture de fenêtre).
        
        Retourne :
            bool : False si l'utilisateur a fermé la fenêtre
        """
        if not self.mode_graphique:
            return True
        
        pygame = self.pygame
        
        # --- Gestion des événements ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # --- Fond noir ---
        self.fenetre.fill(NOIR)
        
        # --- Grille (lignes subtiles) ---
        for x in range(0, self.largeur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.fenetre, GRIS, (x, 0),
                           (x, self.hauteur * TAILLE_CASE))
        for y in range(0, self.hauteur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.fenetre, GRIS, (0, y),
                           (self.largeur * TAILLE_CASE, y))
        
        # --- Nourriture (rouge, avec bordures arrondies) ---
        fx, fy = self.nourriture
        pygame.draw.rect(self.fenetre, ROUGE, (
            fx * TAILLE_CASE + 2, fy * TAILLE_CASE + 2,
            TAILLE_CASE - 4, TAILLE_CASE - 4
        ), border_radius=5)
        
        # --- Corps du serpent ---
        for i, (sx, sy) in enumerate(self.corps):
            # Tête = vert clair, corps = vert foncé
            couleur = VERT_CLAIR if i == 0 else VERT_FONCE
            pygame.draw.rect(self.fenetre, couleur, (
                sx * TAILLE_CASE + 1, sy * TAILLE_CASE + 1,
                TAILLE_CASE - 2, TAILLE_CASE - 2
            ), border_radius=3)
        
        # --- Affichage du score ---
        texte = self.police.render(
            f"Score: {self.score}  |  Steps: {self.steps}", True, BLANC
        )
        self.fenetre.blit(texte, (5, 5))
        
        # --- Mise à jour de l'affichage ---
        pygame.display.flip()
        
        # --- Contrôle de vitesse ---
        if self.vitesse > 0:
            self.horloge.tick(self.vitesse)
        
        return True
    
    def close(self):
        """Ferme proprement Pygame."""
        if self.mode_graphique and self.pygame:
            self.pygame.quit()
