# Fichiers Inclus dans le Répertoire

Ce répertoire contient maintenant **tous les fichiers nécessaires** pour entraîner et tester le Go1 de manière autonome.

## ✅ Fichiers copiés depuis IsaacLab

### Configuration de base (`config/base/`)
- **velocity_env_cfg.py** (330 lignes)
  - Configuration complète de l'environnement de locomotion
  - Définit scène, observations, actions, récompenses, terminaisons, events
  - Imports adaptés pour utiliser les fichiers MDP locaux

### Définitions MDP (`config/mdp/`)
- **rewards.py** (~150 lignes) - Fonctions de récompense:
  - `track_lin_vel_xy_exp` - Suivi de vitesse linéaire
  - `track_ang_vel_z_exp` - Suivi de vitesse angulaire
  - `feet_air_time` - Récompense pour marche naturelle
  - `undesired_contacts` - Pénalité contacts indésirables
  - Pénalités pour couples, accélérations, actions

- **terminations.py** (~50 lignes) - Conditions d'arrêt:
  - `time_out` - Fin d'épisode après 20s
  - `illegal_contact` - Arrêt si chute/contact indésirable
  - `joint_pos_out_of_manual_limit` - Limites articulaires

- **curriculums.py** (~48 lignes) - Apprentissage progressif:
  - `terrain_levels_vel` - Augmente difficulté du terrain

- **__init__.py** - Exports des fonctions MDP

### Configuration Go1 (`config/`)
- **go1_walking_env_cfg.py** - Configuration personnalisée Go1
  - Hérite de `LocomotionVelocityRoughEnvCfg`
  - Adapte terrain, masse, récompenses pour le Go1
  - Version training et play

- **__init__.py** - Enregistrement des tâches Gym:
  - `Isaac-Velocity-Rough-Unitree-Go1-Custom-v0`
  - `Isaac-Velocity-Rough-Unitree-Go1-Custom-Play-v0`

### Configuration agent (`config/agents/`)
- **rsl_rl_ppo_cfg.py** - Hyperparamètres PPO:
  - `UnitreeGo1RoughPPORunnerCfg` - 1500 iterations
  - `UnitreeGo1FlatPPORunnerCfg` - 300 iterations
  - Architecture réseau, learning rate, etc.

## 📁 Structure complète

```
my_go1_project/
├── config/
│   ├── __init__.py                     # Enregistrement Gym
│   ├── go1_walking_env_cfg.py          # Config Go1 spécifique
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rsl_rl_ppo_cfg.py          # Config PPO
│   ├── base/
│   │   ├── __init__.py
│   │   └── velocity_env_cfg.py         # Config base locomotion
│   └── mdp/
│       ├── __init__.py
│       ├── rewards.py                  # Récompenses
│       ├── terminations.py             # Terminaisons
│       └── curriculums.py              # Curriculum
├── scripts/
│   ├── train_go1_walking.py           # Script training
│   └── play_go1_walking.py            # Script test
├── train.sh                            # Raccourci training
├── test.sh                             # Raccourci test
├── README.md                           # Documentation principale
├── PROJECT_STRUCTURE.md                # Détails structure
└── FILES_INCLUDED.md                  # Ce fichier
```

## 🔄 Imports modifiés

Les imports ont été adaptés pour pointer vers les fichiers locaux:

**Avant** (isaaclab_tasks):
```python
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
```

**Après** (local):
```python
from .base.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from mdp import *  # Import local
```

## 📊 Statistiques

- **Total lignes copiées**: ~578 lignes
- **Fichiers Python**: 10 fichiers
- **Dossiers créés**: 3 (base/, mdp/, agents/)
- **Fonctions MDP**: ~30 fonctions (rewards + terminations + curriculum)

## 🎯 Avantages

1. **Autonomie**: Tous les fichiers nécessaires sont dans le repo
2. **Versioning**: Contrôle complet des changements
3. **Personnalisation**: Facile de modifier récompenses/terminaisons
4. **Reproductibilité**: Configuration complète sauvegardée
5. **Documentation**: Structure claire et documentée

## 🚀 Prochaines étapes

Pour personnaliser le comportement du robot:

1. **Modifier les récompenses**: Éditer `config/go1_walking_env_cfg.py`
   - Ajuster les poids (`weight`)
   - Activer/désactiver des termes

2. **Ajouter des récompenses**: Éditer `config/mdp/rewards.py`
   - Créer de nouvelles fonctions
   - Les utiliser dans `go1_walking_env_cfg.py`

3. **Changer les terminaisons**: Éditer `config/mdp/terminations.py`
   - Ajouter de nouvelles conditions d'arrêt

4. **Ajuster l'apprentissage**: Éditer `config/agents/rsl_rl_ppo_cfg.py`
   - Learning rate, architecture réseau, etc.

## ⚠️ Note importante

Les fichiers dans `config/base/` et `config/mdp/` sont des **copies** des fichiers IsaacLab. 
Si IsaacLab est mis à jour, ces fichiers ne seront **pas** automatiquement mis à jour.
Cela permet d'avoir un projet stable et reproductible.
