# -*- coding: utf-8 -*-
"""
=============================================================================
 SNAKE IA — Générateur de Graphe de Benchmark (Échelle Logarithmique)
=============================================================================
 Charge l'historique d'entraînement de models/training_history.json
 et trace le score moyen sur 100 parties en fonction des épisodes (échelle log).
=============================================================================
"""

import os
import json
import matplotlib.pyplot as plt

FICHIER_HISTORIQUE = os.path.join("models", "training_history.json")
IMAGE_BENCHMARK = os.path.join("models", "benchmark.png")

def main():
    if not os.path.exists(FICHIER_HISTORIQUE):
        print(f"[!!] Historique introuvable : {FICHIER_HISTORIQUE}")
        print("Veuillez lancer un entraînement DQN d'abord.")
        return

    try:
        with open(FICHIER_HISTORIQUE, "r") as f:
            data = json.load(f)
            episodes = data.get("episodes", [])
            scores_moyens = data.get("scores_moyens", [])
    except Exception as e:
        print(f"[!!] Erreur lors de la lecture du fichier : {e}")
        return

    if not episodes:
        print("[!!] Historique vide.")
        return

    print(f"[*] Chargement de {len(episodes)} points de mesure...")
    
    # Création de la figure
    plt.figure(figsize=(10, 6))
    
    # Tracé du score moyen
    plt.plot(episodes, scores_moyens, marker='o', color='#00C8FF', linewidth=2, label="Score moyen (100 parties)")
    
    # Configuration des axes
    plt.xscale('log')
    plt.xlabel("Nombre d'épisodes d'entraînement (Échelle Logarithmique)", fontsize=12)
    plt.ylabel("Score Moyen (Moyenne glissante 100)", fontsize=12)
    plt.title("Progression des performances DQN en fonction de l'entraînement", fontsize=14, fontweight='bold')
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    plt.savefig(IMAGE_BENCHMARK, dpi=300)
    plt.close()
    
    print(f"[OK] Graphe de benchmark généré avec succès dans : {IMAGE_BENCHMARK}")

if __name__ == "__main__":
    main()
