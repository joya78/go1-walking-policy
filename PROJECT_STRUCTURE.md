# Go1 Walking Policy - Structure du Projet

## Structure des fichiers

```
my_go1_project/
├── config/
│   ├── __init__.py                     # Enregistrement des tâches Gym
│   ├── go1_walking_env_cfg.py          # Configuration Go1 spécifique
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py          # Configuration PPO pour l'entraînement
│   ├── base/
│   │   ├── __init__.py
│   │   └── velocity_env_cfg.py         # Configuration de base pour locomotion
│   └── mdp/
│       ├── __init__.py
│       ├── rewards.py                  # Fonctions de récompenses
│       ├── terminations.py             # Conditions de terminaison
│       └── curriculums.py              # Curriculum learning
├── scripts/
│   ├── train_go1_walking.py           # Script d'entraînement
│   └── play_go1_walking.py            # Script de test/évaluation
├── train.sh                            # Raccourci pour lancer l'entraînement
├── test.sh                             # Raccourci pour tester la policy
└── videos/                             # Vidéos enregistrées

```

## Fichiers copiés depuis IsaacLab

Ce projet inclut maintenant **tous les fichiers nécessaires** pour être autonome:

### Configuration de base (`config/base/`)
- **velocity_env_cfg.py**: Configuration complète de l'environnement de locomotion
  - Définit la scène (terrain, robot, capteurs)
  - Configure les commandes de vélocité
  - Définit les observations, actions, récompenses, terminaisons
  - Gère les événements (perturbations, randomisation)

### Fonctions MDP (`config/mdp/`)
- **rewards.py**: Toutes les fonctions de récompense
  - Suivi de trajectoire (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`)
  - Pénalités (mouvement vertical, couples, accélérations)
  - Récompenses de marche (temps en l'air des pieds, contacts indésirables)
  
- **terminations.py**: Conditions de fin d'épisode
  - Timeout
  - Contact avec le sol (chute)
  - Limites articulaires
  
- **curriculums.py**: Apprentissage progressif
  - Niveaux de terrain adaptatifs

## Configuration Go1 personnalisée

Le fichier `config/go1_walking_env_cfg.py` hérite de la configuration de base et personnalise:

- **Échelle du terrain**: Adapté à la taille du Go1
- **Échelle d'action**: Réduite à 0.25 pour des mouvements plus doux
- **Masse ajoutée**: Randomisation entre -1kg et +3kg
- **Poids des récompenses**: Optimisés pour le Go1
- **Contacts**: Configuration pour les pieds ".*_foot" et corps "trunk"

## Différence entre Entraînement et Test

### Entraînement (Isaac-Velocity-Rough-Unitree-Go1-Custom-v0)
- **Environnements**: 4096 robots en parallèle
- **Curriculum**: Terrain de difficulté progressive
- **Perturbations**: Randomisation de masse, friction, poussées
- **Corruption**: Bruit sur les observations

### Test (Isaac-Velocity-Rough-Unitree-Go1-Custom-Play-v0)
- **Environnements**: 50 robots
- **Curriculum**: Désactivé (terrains variés aléatoires)
- **Perturbations**: Désactivées
- **Corruption**: Désactivée

## Paramètres temporels

- **dt simulation**: 0.005s (200 Hz)
- **decimation**: 4 → action à 50 Hz (0.02s)
- **episode_length_s**: 20s
- **num_steps_per_env**: 24 steps par itération de training
- **max_iterations**: 1500 itérations (ou 300 pour flat)

## Utilisation

### Entraînement
```bash
bash train.sh
```

### Test
```bash
bash test.sh --checkpoint logs/rsl_rl/unitree_go1_rough/RUN/model_XXX.pt
```

### Avec enregistrement vidéo
```bash
bash test.sh --checkpoint MODEL.pt --video
```

## Dépendances

Ce projet nécessite:
- Isaac Sim / IsaacLab
- RSL-RL (bibliothèque RL)
- isaaclab_assets (pour UNITREE_GO1_CFG)
- PyTorch

## Notes importantes

- Les fichiers dans `config/base/` et `config/mdp/` sont des **copies** des fichiers IsaacLab
- Ils permettent au projet d'être autonome et versionné
- Les modifications de récompenses/terminaisons doivent être faites dans `config/go1_walking_env_cfg.py`
- Les imports ont été adaptés pour pointer vers les fichiers locaux
