#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour analyser le nombre optimal de clusters en utilisant la méthode
de la silhouette sur l'ensemble des retours de service.
Version améliorée avec exclusion d'outliers et analyse multi-critères.
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Ajout du chemin des modules
chemin_dossier_parent = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(chemin_dossier_parent)


def analyze_silhouette(name, df, z_threshold=3.0):
    """Analyse les clusters optimaux en utilisant plusieurs métriques de validation."""
    # 1) Conversion et suppression des NaN
    points = df.astype(float).to_numpy()
    mask = ~np.isnan(points).any(axis=1)
    points = points[mask]

    if len(points) == 0:
        print("Aucun point trouvé")
        return

    # 2) Inversion des signes pour points en Y négatif (logique existante)
    for i in range(len(points)):
        if points[i, 1] < 0:
            points[i, 0] *= -1
            points[i, 1] *= -1

    # 3) Première normalisation pour la détection des outliers
    scaler_outliers = StandardScaler()
    points_scaled_for_outliers = scaler_outliers.fit_transform(points)

    # 4) Exclusion des outliers par Z‑score (|z| > z_threshold)
    mask_inliers = np.all(
        np.abs(points_scaled_for_outliers) <= z_threshold, axis=1
    )
    points_clean = points[mask_inliers]

    print(
        f"Points initiaux : {len(points)}, points après exclusion outliers : {len(points_clean)}"
    )

    if len(points_clean) == 0:
        print("Aucun point restant après exclusion des outliers")
        return

    # 5) Normalisation finale sur les données nettoyées
    scaler_final = StandardScaler()
    points_scaled = scaler_final.fit_transform(points_clean)

    # 6) Calculer plusieurs métriques pour différentes valeurs de k
    k_range = range(2, 8)  # Élargi la plage
    silhouette_scores = []
    calinski_harabasz_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(points_scaled)

        # Métriques de validation
        sil_score = silhouette_score(points_scaled, clusters)
        ch_score = calinski_harabasz_score(points_scaled, clusters)

        silhouette_scores.append(sil_score)
        calinski_harabasz_scores.append(ch_score)
        inertias.append(kmeans.inertia_)

    # 7) Tracer les trois métriques
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Silhouette
    axes[0].plot(k_range, silhouette_scores, "bx-")
    axes[0].set_xlabel("k (nombre de clusters)")
    axes[0].set_ylabel("Score de silhouette moyen")
    axes[0].set_title("Scores de silhouette")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Calinski-Harabasz
    axes[1].plot(k_range, calinski_harabasz_scores, "gx-")
    axes[1].set_xlabel("k (nombre de clusters)")
    axes[1].set_ylabel("Score Calinski-Harabasz")
    axes[1].set_title("Scores Calinski-Harabasz")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # Inertie (pour comparaison avec méthode du coude)
    axes[2].plot(k_range, inertias, "rx-")
    axes[2].set_xlabel("k (nombre de clusters)")
    axes[2].set_ylabel("Inertie")
    axes[2].set_title("Inertie (méthode du coude)")
    axes[2].grid(True, linestyle="--", alpha=0.7)

    # 8) Déterminer le k optimal selon chaque méthode
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_ch = k_range[np.argmax(calinski_harabasz_scores)]

    # Marquer les optimums
    axes[0].plot(
        optimal_k_silhouette,
        silhouette_scores[optimal_k_silhouette - 2],
        "ro",
        markersize=10,
        label=f"Optimal (k={optimal_k_silhouette})",
    )
    axes[1].plot(
        optimal_k_ch,
        calinski_harabasz_scores[optimal_k_ch - 2],
        "ro",
        markersize=10,
        label=f"Optimal (k={optimal_k_ch})",
    )

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"images/multi_criteria_analysis_{name.replace(' ', '_').lower()}.svg"
    )
    plt.close()

    # 9) Analyse des résultats et choix final
    print(f"\nAnalyse multi-critères pour {name}:")
    print(
        f"Silhouette suggère    : k={optimal_k_silhouette} (score: {max(silhouette_scores):.3f})"
    )
    print(
        f"Calinski-Harabasz suggère : k={optimal_k_ch} (score: {max(calinski_harabasz_scores):.1f})"
    )

    # Logique de décision transparente
    if optimal_k_silhouette == optimal_k_ch:
        final_k = optimal_k_silhouette
        print(f"✓ Consensus : k={final_k}")
    else:
        print("⚠ Désaccord entre les métriques")

        # Analyser les scores pour k=3 et k=4 spécifiquement
        sil_k3 = (
            silhouette_scores[1] if 3 in k_range else 0
        )  # index 1 car k_range commence à 2
        sil_k4 = silhouette_scores[2] if 4 in k_range else 0  # index 2
        ch_k3 = calinski_harabasz_scores[1] if 3 in k_range else 0
        ch_k4 = calinski_harabasz_scores[2] if 4 in k_range else 0

        print(f"Comparaison k=3 vs k=4:")
        print(
            f"  k=3 : Silhouette={sil_k3:.3f}, Calinski-Harabasz={ch_k3:.1f}"
        )
        print(
            f"  k=4 : Silhouette={sil_k4:.3f}, Calinski-Harabasz={ch_k4:.1f}"
        )

        # Décision basée sur la différence relative
        sil_diff = abs(sil_k4 - sil_k3) / max(sil_k3, sil_k4)

        if sil_diff < 0.05:  # Si différence < 5%
            final_k = 4  # Préférer k=4 pour l'interprétation tennis
            print(
                f"→ Différence silhouette faible ({sil_diff:.3f}), choix k=4 pour cohérence métier"
            )
        else:
            final_k = optimal_k_silhouette
            print(
                f"→ Différence silhouette significative, choix silhouette : k={final_k}"
            )

    print(f"Choix final : k={final_k}")
    print(f"Nombre total de points utilisés : {len(points_clean)}")

    # 10) Clustering final et visualisation
    optimal_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    clusters = optimal_kmeans.fit_predict(points_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        points_clean[:, 0], points_clean[:, 1], c=clusters, cmap="viridis"
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title(
        f"Distribution des points en {final_k} clusters (méthode multi-critères)"
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(
        f"images/silhouette_distribution_{name.replace(' ', '_').lower()}.svg"
    )
    plt.close()

    # 11) Statistiques détaillées des clusters
    silhouette_values = silhouette_samples(points_scaled, clusters)
    print("\nStatistiques des clusters:")
    for i in range(final_k):
        cluster_points = points_clean[clusters == i]
        cluster_silhouette = silhouette_values[clusters == i]

        print(f"\nCluster {i+1}:")
        print(f"  Nombre de points : {len(cluster_points)}")
        print(
            f"  Centre     : ({np.mean(cluster_points[:, 0]):.2f}, {np.mean(cluster_points[:, 1]):.2f})"
        )
        print(
            f"  Écart‑type : ({np.std(cluster_points[:, 0]):.2f}, {np.std(cluster_points[:, 1]):.2f})"
        )
        print(f"  Score silhouette moyen : {np.mean(cluster_silhouette):.3f}")


def main():
    """Fonction principale."""
    DB_PATH = "BDD_avec_cluster.db"
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT coor_balle_x, coor_balle_y
    FROM Liste_des_coups
    WHERE coor_balle_x IS NOT NULL 
      AND coor_balle_y IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    analyze_silhouette("Tous les retours de service", df, z_threshold=2.5)

    conn.close()


if __name__ == "__main__":
    main()
