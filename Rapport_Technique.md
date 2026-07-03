# Modélisation Mathématique & Logique Technique du Projet Snake IA

Ce document décrit en détail les fondements mathématiques et la logique logicielle qui régissent l'IA développée pour ce projet, optimisé pour l'Orange Pi 3B.

---

## 1. Processus de Décision Markovien (MDP)

Le jeu Snake est modélisé comme un **Processus de Décision Markovien** défini par le quintuplet $(S, A, P, R, \gamma)$ :

*   **Espace d'États $S$ (Dimension 11)** :
    L'état $s_t \in S$ perçu à l'instant $t$ est un vecteur combinant :
    1.  *Danger immédiat* : $s_t[0..2] \in \{0, 1\}^3$ (Devant, Gauche, Droite relative).
    2.  *Direction absolue* : $s_t[3..6] \in \{0, 1\}^4$ (One-Hot de l'orientation de la tête).
    3.  *Nourriture relative* : $s_t[7..10] \in \{0, 1\}^4$ (Devant, Derrière, Gauche, Droite relative par rapport à la tête et son vecteur de marche).
*   **Espace d'Actions $A$ (Dimension 3)** :
    L'agent interagit uniquement avec des actions relatives :
    $$A = \{0 \text{ (Tout droit)}, 1 \text{ (Tourner à Gauche)}, 2 \text{ (Tourner à Droite)}\}$$
*   **Fonction de Transition $P(s_{t+1} \mid s_t, a_t)$** :
    Déterministe. Elle déplace le serpent d'une case sur la grille en fonction de l'action choisie.
*   **Fonction de Récompense $R(s_t, a_t)$** :
    Formulée avec du *reward shaping* pour accélérer la convergence :
    $$R(s_t, a_t) = \begin{cases} 
      +10.0 & \text{si le serpent mange la nourriture} \\
      -10.0 & \text{si le serpent entre en collision (mort)} \\
      +1.0  & \text{si la distance de Manhattan } d_M(\text{tête}, \text{nourriture}) \text{ diminue} \\
      -1.0  & \text{si la distance de Manhattan } d_M(\text{tête}, \text{nourriture}) \text{ augmente} \\
      0.0   & \text{sinon}
    \end{cases}$$
*   **Facteur d'atténuation $\gamma = 0.99$** :
    Pondère l'importance des récompenses futures.

---

## 2. Apprentissage par Imitation (Behavioral Cloning)

L'apprentissage par imitation utilise une méthode d'apprentissage supervisé. À partir des données humaines collectées $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^N$ :

### Logique Mathématique
Le réseau $\pi_\theta(s)$ paramétré par ses poids $\theta$ cherche à prédire la distribution de probabilité des actions possibles. On minimise la perte de **l'Entropie Croisée (Cross-Entropy Loss)** :

$$\mathcal{L}_{BC}(\theta) = - \frac{1}{B} \sum_{i=1}^B \sum_{c=0}^{2} y_{i,c} \log \left( \text{softmax}(\pi_\theta(s_i))_c \right)$$

Où $B$ est la taille du mini-batch, et $y_{i,c}$ vaut 1 si $c = a_i$ (l'action humaine), 0 sinon.

---

## 3. Apprentissage par Renforcement : Deep Q-Learning (DQN)

Le but est d'apprendre la fonction de valeur optimale Action-Valeur $Q^*(s, a)$, représentant l'espérance des récompenses futures cumulées actualisées.

### Équation de Bellman
La fonction $Q$ optimale satisfait la relation de récurrence de Bellman :
$$Q^*(s, y) = R(s, a) + \gamma \max_{a'} Q^*(s', a')$$

Dans le Deep Q-Learning, nous approximons $Q^*(s, a)$ par un réseau de neurones $Q(s, a; \theta_i)$.

### Réseau Cible (Target Network) et Loss
Pour stabiliser l'entraînement (éviter que la cible ne change à chaque pas de gradient), nous utilisons un réseau cible distinct avec ses propres paramètres $\theta^-$. La perte quadratique moyenne (MSE) minimisée à chaque étape d'apprentissage est :

$$\mathcal{L}_{DQN}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( y_i - Q(s_i, a_i; \theta) \right)^2$$

Où la cible $y_i$ (TD-target) est définie par :
$$y_i = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \theta^-)$$

($d_i = 1$ si l'état $s'_i$ est terminal, 0 sinon).

### Mise à jour des poids
*   Le réseau principal $\theta$ est mis à jour par rétropropagation du gradient de $\mathcal{L}_{DQN}(\theta)$.
*   Le réseau cible $\theta^-$ est synchronisé périodiquement (tous les $C=10$ épisodes) : $\theta^- \leftarrow \theta$.

---

## 4. Politique d'Exploration : $\epsilon$-greedy

Afin d'équilibrer l'exploration (découvrir de nouvelles stratégies) et l'exploitation (utiliser les connaissances acquises), l'action $a_t$ est sélectionnée selon :

$$a_t = \begin{cases}
  \text{action aléatoire dans } A & \text{avec probabilité } \epsilon \\
  \arg\max_{a \in A} Q(s_t, a; \theta) & \text{avec probabilité } 1 - \epsilon
\end{cases}$$

Avec une décroissance géométrique à chaque épisode :
$$\epsilon_{t+1} = \max(\epsilon_{fin}, \epsilon_t \times \lambda)$$
Où $\lambda = 0.995$ et $\epsilon_{fin} = 0.01$.
