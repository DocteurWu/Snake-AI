# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Entraînement DQN Multi-Agent Parallèle avec Graphique et Slider
=============================================================================
 Fait tourner 7 serpents en parallèle dans une grille Pygame (2x4).
 La 8ème case affiche l'historique des scores et un Slider interactif 
 pour contrôler finement la vitesse de calcul (de 5 à 1000 FPS ou mode Max).
 
 Contrôles :
   Clic souris : Ajuster la vitesse avec le Slider dans la 8ème case
   ESPACE      : Mettre l'entraînement en pause / reprise
   S           : Sauvegarder manuellement le modèle
   ECHAP       : Sauvegarder et quitter
=============================================================================
"""

import os
import sys
import time
import pygame
import numpy as np

from config import (
    TAILLE_GRILLE, TAILLE_CASE, VITESSE_IA,
    NOIR, BLANC, ROUGE, VERT_FONCE, VERT_CLAIR, GRIS, BLEU,
    DROITE, GAUCHE, HAUT, BAS, SENS_HORAIRE,
    TAILLE_ETAT, NB_ACTIONS, NOM_MODELE_DQN
)
from game import SnakeGame
from agent_dqn import AgentDQN

# Dimensions de la fenêtre globale
LARGEUR_SOUS_JEU = TAILLE_GRILLE * TAILLE_CASE # 400 px
HAUTEUR_SOUS_JEU = TAILLE_GRILLE * TAILLE_CASE # 400 px
NB_COLONNES = 4
NB_LIGNES = 2

LARGEUR_FENETRE = LARGEUR_SOUS_JEU * NB_COLONNES # 1600 px
HAUTEUR_FENETRE = HAUTEUR_SOUS_JEU * NB_LIGNES   # 800 px

class SubSnakeGame(SnakeGame):
    """Instance de jeu Snake dessinant sur sa propre sous-surface."""
    def __init__(self, surface):
        self.mode_graphique = True
        self.largeur = TAILLE_GRILLE
        self.hauteur = TAILLE_GRILLE
        self.surface = surface
        self.pygame = pygame
        self.police = pygame.font.SysFont('arial', 14)
        self.vitesse = 0
        self.reset()

    def render(self, id_jeu=0):
        self.surface.fill(NOIR)
        # Grille
        for x in range(0, self.largeur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.surface, GRIS, (x, 0), (x, self.hauteur * TAILLE_CASE))
        for y in range(0, self.hauteur * TAILLE_CASE, TAILLE_CASE):
            pygame.draw.line(self.surface, GRIS, (0, y), (self.largeur * TAILLE_CASE, y))
            
        # Nourriture
        fx, fy = self.nourriture
        pygame.draw.rect(self.surface, ROUGE, (
            fx * TAILLE_CASE + 2, fy * TAILLE_CASE + 2,
            TAILLE_CASE - 4, TAILLE_CASE - 4
        ), border_radius=5)
        
        # Serpent
        for i, (sx, sy) in enumerate(self.corps):
            couleur = VERT_CLAIR if i == 0 else VERT_FONCE
            pygame.draw.rect(self.surface, couleur, (
                sx * TAILLE_CASE + 1, sy * TAILLE_CASE + 1,
                TAILLE_CASE - 2, TAILLE_CASE - 2
            ), border_radius=3)
            
        txt = self.police.render(f"IA #{id_jeu} | Score: {self.score}", True, BLANC)
        self.surface.blit(txt, (5, 5))


class Slider:
    """Un curseur interactif dessiné en Pygame pour régler les FPS."""
    def __init__(self, x, y, largeur, hauteur, val_min, val_max, val_init):
        self.rect = pygame.Rect(x, y, largeur, hauteur)
        self.min = val_min
        self.max = val_max
        self.valeur = val_init
        self.bouton_rect = pygame.Rect(x, y - 5, 10, hauteur + 10)
        self.update_bouton_pos()
        self.drag = False
        self.mode_max = False # Vitesse illimitée

    def update_bouton_pos(self):
        ratio = (self.valeur - self.min) / (self.max - self.min)
        self.bouton_rect.x = self.rect.x + int(ratio * (self.rect.width - self.bouton_rect.width))

    def draw(self, surface):
        police = pygame.font.SysFont('arial', 13)
        
        # Dessiner le rail du slider
        pygame.draw.rect(surface, GRIS, self.rect, border_radius=3)
        
        # Remplissage de la valeur courante
        largeur_remplissage = self.bouton_rect.centerx - self.rect.x
        if largeur_remplissage > 0:
            remplissage_rect = pygame.Rect(self.rect.x, self.rect.y, largeur_remplissage, self.rect.height)
            pygame.draw.rect(surface, BLEU, remplissage_rect, border_radius=3)

        # Dessiner le curseur
        couleur_curseur = BLANC if not self.drag else VERT_CLAIR
        pygame.draw.rect(surface, couleur_curseur, self.bouton_rect, border_radius=2)
        
        # Texte informatif
        val_txt = "MAX (No limit)" if self.mode_max else f"{int(self.valeur)} FPS"
        txt = police.render(f"Vitesse : {val_txt}", True, BLANC)
        surface.blit(txt, (self.rect.x, self.rect.y - 20))

    def handle_event(self, event, offset_x, offset_y):
        """Prend en compte les clics et le glisser-déposer relatif aux coordonnées globales."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Ajuster le clic avec l'offset de la sous-case 8
                pos = (event.pos[0] - offset_x, event.pos[1] - offset_y)
                if self.bouton_rect.collidepoint(pos) or self.rect.collidepoint(pos):
                    self.drag = True
                    self.mode_max = False
                    self.update_valeur(pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.drag = False
        elif event.type == pygame.MOUSEMOTION:
            if self.drag:
                pos_x = event.pos[0] - offset_x
                self.update_valeur(pos_x)

    def update_valeur(self, pos_x):
        # Borner x dans les limites du rail
        x_relatif = max(self.rect.x, min(pos_x, self.rect.x + self.rect.width - self.bouton_rect.width))
        ratio = (x_relatif - self.rect.x) / (self.rect.width - self.bouton_rect.width)
        self.valeur = self.min + ratio * (self.max - self.min)
        
        # Seuil haut pour passer en vitesse "Max"
        if self.valeur >= self.max - 20:
            self.mode_max = True
        else:
            self.mode_max = False
            
        self.update_bouton_pos()


def dessiner_graphe(surface, scores, moyennes, slider):
    """Dessine le graphe d'évolution et le curseur de vitesse dans la case 8."""
    surface.fill((20, 20, 20))
    pygame.draw.rect(surface, BLEU, (0, 0, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU), 2)
    
    # Rendu du Slider interactif
    slider.draw(surface)
    
    police = pygame.font.SysFont('arial', 13)
    
    # Boutons de présélections rapides
    b_50 = police.render("[50 FPS]", True, VERT_CLAIR)
    b_max = police.render("[Vitesse MAX]", True, ROUGE)
    surface.blit(b_50, (30, 75))
    surface.blit(b_max, (140, 75))
    
    # Affichage du graphe
    margin_x, margin_y = 50, 40
    g_w = LARGEUR_SOUS_JEU - margin_x - 20
    g_h = HAUTEUR_SOUS_JEU - margin_y - 140 # Laisser de la place pour le slider en haut
    
    # Décalage de la zone graphique vers le bas
    base_y = HAUTEUR_SOUS_JEU - margin_y
    top_y = 130
    
    # Axes
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (LARGEUR_SOUS_JEU - 20, base_y), 1)
    pygame.draw.line(surface, BLANC, (margin_x, base_y), (margin_x, top_y), 1)
    
    if len(scores) < 2:
        txt = police.render("En attente de donnees...", True, GRIS)
        surface.blit(txt, (margin_x + 50, top_y + g_h // 2))
        return
        
    max_score = max(max(scores), 1)
    nb_pts = len(scores)
    
    pts_scores = []
    pts_moyennes = []
    
    for i in range(nb_pts):
        x = margin_x + int((i / (nb_pts - 1)) * g_w)
        y_s = base_y - int((scores[i] / max_score) * g_h)
        pts_scores.append((x, y_s))
        
        y_m = base_y - int((moyennes[i] / max_score) * g_h)
        pts_moyennes.append((x, y_m))
        
    if len(pts_scores) > 1:
        pygame.draw.lines(surface, (100, 100, 100), False, pts_scores, 1)
        pygame.draw.lines(surface, VERT_CLAIR, False, pts_moyennes, 2)
        
    # Stats
    lbl_max = police.render(f"Meilleur: {max_score}", True, ROUGE)
    surface.blit(lbl_max, (300, 10))
    lbl_avg = police.render(f"Moy(100): {moyennes[-1]:.1f}", True, VERT_CLAIR)
    surface.blit(lbl_avg, (300, 28))
    lbl_partie = police.render(f"Parties: {len(scores)}", True, BLANC)
    surface.blit(lbl_partie, (300, 46))


def main():
    pygame.init()
    fenetre_globale = pygame.display.set_mode((LARGEUR_FENETRE, HAUTEUR_FENETRE))
    pygame.display.set_caption("🐍 Snake IA — Entraînement RL Multi-Agent & Slider interactif")
    horloge = pygame.time.Clock()
    
    # Création des 8 sous-surfaces
    surfaces = []
    for lig in range(NB_LIGNES):
        for col in range(NB_COLONNES):
            rect = pygame.Rect(col * LARGEUR_SOUS_JEU, lig * HAUTEUR_SOUS_JEU, LARGEUR_SOUS_JEU, HAUTEUR_SOUS_JEU)
            surfaces.append(fenetre_globale.subsurface(rect))
            
    # Coordonnées de départ de la case 8 pour l'offset de la souris
    offset_x = 3 * LARGEUR_SOUS_JEU
    offset_y = 1 * HAUTEUR_SOUS_JEU
    
    # Case #8 Slider: x=30, y=40, largeur=240, hauteur=12, min=5, max=1000, init=60
    slider = Slider(30, 40, 240, 12, 5, 1000, 60)
    
    # Initialisation des 7 jeux (cases 1 à 7)
    jeux = [SubSnakeGame(surfaces[i]) for i in range(7)]
    agent = AgentDQN()
    
    if os.path.exists(os.path.join("models", NOM_MODELE_DQN)):
        try:
            agent.charger()
        except Exception:
            pass

    scores_historique = []
    moyennes_historique = []
    liste_scores_recents = []
    
    en_pause = False
    en_cours = True
    
    etats = [jeu.get_state() for jeu in jeux]
    
    print("\nEntraînement multi-agents avec Slider lancé !")
    
    while en_cours:
        # Vitesse courante dictée par le slider
        fps_ia = 0 if slider.mode_max else int(slider.valeur)

        # --- Événements globaux et souris ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    en_cours = False
                elif event.key == pygame.K_SPACE:
                    en_pause = not en_pause
                    print(f"Pause : {en_pause}")
                elif event.key == pygame.K_s:
                    agent.sauvegarder()
            
            # Passer les coordonnées de souris au Slider avec offset
            slider.handle_event(event, offset_x, offset_y)
            
            # Gestion des clics sur les boutons rapides
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = (event.pos[0] - offset_x, event.pos[1] - offset_y)
                # Clic bouton "[50 FPS]"
                if 30 <= pos[0] <= 110 and 70 <= pos[1] <= 95:
                    slider.valeur = 50
                    slider.mode_max = False
                    slider.update_bouton_pos()
                # Clic bouton "[Vitesse MAX]"
                elif 140 <= pos[0] <= 240 and 70 <= pos[1] <= 95:
                    slider.mode_max = True
                    slider.valeur = slider.max
                    slider.update_bouton_pos()

        if en_pause:
            pygame.time.wait(100)
            continue
            
        # --- Une étape pour chaque serpent ---
        for i, jeu in enumerate(jeux):
            action = agent.choisir_action(etats[i], entrainement=True)
            etat_suivant, recompense, termine, score = jeu.step(action)
            agent.memoriser(etats[i], action, recompense, etat_suivant, termine)
            agent.entrainer()
            jeu.render(id_jeu=i+1)
            etats[i] = etat_suivant
            
            if termine:
                scores_historique.append(score)
                liste_scores_recents.append(score)
                if len(liste_scores_recents) > 100:
                    liste_scores_recents.pop(0)
                moyennes_historique.append(sum(liste_scores_recents) / len(liste_scores_recents))
                
                etats[i] = jeu.reset()
                agent.fin_episode()
                
                if len(scores_historique) % 50 == 0:
                    agent.sauvegarder()

        # Dessiner le graphe et le slider dans la case 8
        dessiner_graphe(surfaces[7], scores_historique, moyennes_historique, slider)
        
        # Mettre à jour l'écran global
        pygame.display.flip()
        
        # Si on n'est pas en mode MAX, on limite la vitesse de rendu
        if fps_ia > 0:
            horloge.tick(fps_ia)
            
    # Sauvegarde finale
    agent.sauvegarder()
    pygame.quit()
    print("Entraînement terminé proprement.")

if __name__ == "__main__":
    main()
