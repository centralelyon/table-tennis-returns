#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour analyser le nombre optimal de clusters en utilisant la méthode du coude
sur l'ensemble des retours de service, en excluant d'abord les outliers (|z| > 3).
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ajout du chemin des modules
chemin_dossier_parent = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(chemin_dossier_parent)


def analyze_clusters(name, df, z_threshold=3.0):
    """Analyse les clusters optimaux pour un ensemble de points en excluant d'abord les outliers."""
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

    # 3) Normalisation
    scaler = StandardScaler()
    points_scaled_full = scaler.fit_transform(points)

    # 4) Exclusion des outliers par Z‑score (|z| > z_threshold)
    mask_inliers = np.all(np.abs(points_scaled_full) <= z_threshold, axis=1)
    points = points[mask_inliers]
    points_scaled = points_scaled_full[mask_inliers]

    print(
        f"Points initiaux : {mask_inliers.size}, points après exclusion outliers : {len(points)}"
    )

    # 5) Calcul des inerties pour k=1..6
    k_range = list(range(1, 11))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(points_scaled)
        inertias.append(km.inertia_)

    # 6) Détection du coude via kneed
    kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
    elbow_point = (
        kl.knee
        or k_range[
            np.argmax(np.abs(np.gradient(np.gradient(np.array(inertias)))))
        ]
    )

    # 8) Tracé de la courbe du coude
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, "bx-")
    plt.plot(
        elbow_point,
        inertias[elbow_point - 1],
        "ro",
        markersize=10,
        label=f"Point de coude (k={elbow_point})",
    )
    plt.xlabel("k (nombre de clusters)")
    plt.ylabel("Inertie")
    plt.title(f"Méthode du coude pour {name}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(f"images/elbow_method_{name.replace(' ', '_').lower()}.svg")
    plt.close()

    print(f"\nAnalyse pour {name}:")
    print(f"Nombre optimal de clusters suggéré : {elbow_point}")
    print(f"Nombre total de points utilisés : {len(points)}")

    # 9) Clustering final et visualisation
    optimal_kmeans = KMeans(n_clusters=elbow_point, random_state=42, n_init=10)
    clusters = optimal_kmeans.fit_predict(points_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        points[:, 0], points[:, 1], c=clusters, cmap="viridis"
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title(f"Distribution des points en {elbow_point} clusters")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(
        f"images/clusters_distribution_{name.replace(' ', '_').lower()}.svg"
    )
    plt.close()

    # 10) Statistiques des clusters
    print("\nStatistiques des clusters:")
    for i in range(elbow_point):
        cluster_pts = points[clusters == i]
        print(f"\nCluster {i+1}:")
        print(f"  Nombre de points : {len(cluster_pts)}")
        print(
            f"  Centre     : ({np.mean(cluster_pts[:, 0]):.2f}, {np.mean(cluster_pts[:, 1]):.2f})"
        )
        print(
            f"  Écart‑type : ({np.std(cluster_pts[:, 0]):.2f}, {np.std(cluster_pts[:, 1]):.2f})"
        )


def main():
    """Fonction principale."""
    DB_PATH = "BDD_avec_cluster.db"
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT coor_balle_x, coor_balle_y
    FROM Liste_des_coups
    WHERE coor_balle_x IS NOT NULL 
      AND coor_balle_y IS NOT NULL
      AND num_coup = 2
    """

    df = pd.read_sql_query(query, conn)
    analyze_clusters("Tous les retours de service", df, z_threshold=3.0)

    conn.close()


if __name__ == "__main__":
    main()
