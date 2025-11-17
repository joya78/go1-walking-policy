#!/usr/bin/env python3
"""Generate comprehensive training visualizations with explanations."""

import sys
sys.path.insert(0, '/home/maxime/my_go1_project')

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_FILE = '/home/maxime/IsaacLab-main/logs/rsl_rl/unitree_go1_rough/2025-11-16_17-17-58/events.out.tfevents.1763342289.HR5.174392.0'
OUTPUT_DIR = '/home/maxime/my_go1_project/videos'

def plot_comprehensive_metrics():
    """Plot all important training metrics with explanations."""
    
    ea = event_accumulator.EventAccumulator(LOG_FILE)
    ea.Reload()
    tags = ea.Tags()['scalars']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define metrics to plot with explanations
    metrics_config = [
        # Row 1: Performance globale
        ('Train/mean_reward', '🎯 Récompense Moyenne', 
         'Plus haut = meilleure performance\nCombine tous les termes de récompense'),
        ('Train/mean_episode_length', '⏱️ Durée des Épisodes', 
         'Plus haut = robot reste debout plus longtemps\nMax = 1000 steps (20s)'),
        ('Curriculum/terrain_levels', '🏔️ Niveau de Terrain', 
         'Difficulté du terrain\nAugmente automatiquement si robot réussit'),
        
        # Row 2: Récompenses de tracking
        ('Episode_Reward/track_lin_vel_xy_exp', '🏃 Suivi Vitesse Linéaire', 
         'À quel point le robot suit la vitesse demandée\nPlus haut = meilleur suivi'),
        ('Episode_Reward/track_ang_vel_z_exp', '🔄 Suivi Vitesse Angulaire', 
         'Suivi de la rotation (yaw)\nPlus haut = tourne comme demandé'),
        ('Episode_Reward/feet_air_time', '👣 Temps en l\'Air des Pieds', 
         'Récompense pour marche naturelle\nPlus haut = meilleure démarche'),
        
        # Row 3: Pénalités
        ('Episode_Reward/lin_vel_z_l2', '⬇️ Pénalité Mouvement Vertical', 
         'Négatif = pénalise les sauts\nProche de 0 = mouvement fluide'),
        ('Episode_Reward/ang_vel_xy_l2', '⚖️ Pénalité Roll/Pitch', 
         'Négatif = pénalise inclinaison\nProche de 0 = reste à plat'),
        ('Episode_Reward/dof_torques_l2', '⚡ Pénalité Couples', 
         'Négatif = pénalise effort élevé\nProche de 0 = économe en énergie'),
        
        # Row 4: Loss et learning
        ('Loss/value_function', '📉 Loss Fonction Valeur', 
         'Erreur du critique\nDoit diminuer pendant l\'entraînement'),
        ('Loss/surrogate', '🎭 Loss Policy PPO', 
         'Erreur de la policy\nAutour de 0 = bonnes mises à jour'),
        ('Loss/learning_rate', '📊 Learning Rate', 
         'Taux d\'apprentissage adaptatif\nDiminue si KL trop élevé'),
        
        # Row 5: Performance
        ('Perf/total_fps', '⚡ FPS Total', 
         'Vitesse de simulation\nPlus haut = entraînement plus rapide'),
        ('Metrics/base_velocity/error_vel_xy', '📏 Erreur Vitesse XY', 
         'Écart entre vitesse demandée et réelle\nPlus bas = meilleur contrôle'),
        ('Episode_Termination/time_out', '✅ Épisodes Complets', 
         'Proportion d\'épisodes terminés normalement\n1.0 = aucune chute'),
    ]
    
    n_rows = 5
    n_cols = 3
    
    for idx, (metric, title, description) in enumerate(metrics_config, 1):
        if metric in tags:
            ax = fig.add_subplot(n_rows, n_cols, idx)
            
            events = ea.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            # Plot
            ax.plot(steps, values, linewidth=2)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add description as text
            ax.text(0.02, 0.98, description, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Add current value
            if values:
                current_val = values[-1]
                ax.text(0.98, 0.02, f'Dernier: {current_val:.3f}', 
                       transform=ax.transAxes, fontsize=8,
                       horizontalalignment='right', verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    fig.suptitle('Go1 Training - Vue Complète des Métriques', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = os.path.join(OUTPUT_DIR, 'training_metrics_detailed.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphiques détaillés sauvegardés: {output_path}")
    
    return fig

def print_summary():
    """Print text summary of training."""
    ea = event_accumulator.EventAccumulator(LOG_FILE)
    ea.Reload()
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE L'ENTRAÎNEMENT Go1")
    print("="*60 + "\n")
    
    # Get key metrics
    metrics_to_show = {
        'Train/mean_reward': 'Récompense Moyenne',
        'Train/mean_episode_length': 'Durée Épisode',
        'Episode_Reward/track_lin_vel_xy_exp': 'Suivi Vitesse',
        'Loss/value_function': 'Loss Critique',
        'Perf/total_fps': 'FPS',
    }
    
    for metric, name in metrics_to_show.items():
        try:
            events = ea.Scalars(metric)
            if events:
                initial = events[0].value
                final = events[-1].value
                change = ((final - initial) / abs(initial) * 100) if initial != 0 else 0
                
                emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"{emoji} {name:30s}: {initial:8.3f} → {final:8.3f} ({change:+6.1f}%)")
        except:
            pass
    
    print("\n" + "="*60)
    print("💡 INTERPRÉTATION:")
    print("="*60)
    print("  • Récompense ↗️  = Robot apprend à mieux marcher")
    print("  • Durée ↗️       = Robot reste debout plus longtemps")
    print("  • Loss ↘️        = Modèle converge")
    print("  • Suivi ↗️       = Meilleur contrôle de vitesse")
    print("="*60 + "\n")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🎨 Génération des graphiques détaillés...")
    plot_comprehensive_metrics()
    
    print_summary()
    
    print("\n📁 Fichiers générés:")
    print(f"  • {OUTPUT_DIR}/training_metrics.png (aperçu)")
    print(f"  • {OUTPUT_DIR}/training_metrics_detailed.png (détaillé)")
    print("\n🌐 Pour voir en live: http://localhost:6006 (avec tunnel SSH)")
