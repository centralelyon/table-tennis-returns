#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour analyser le nombre optimal de clusters en utilisant la méthode
de la silhouette sur l'ensemble des retours de service.
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

# Ajout du chemin des modules
chemin_dossier_parent = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(chemin_dossier_parent)


def analyze_silhouette(name, df):
    """Analyse les clusters optimaux en utilisant la méthode de la silhouette."""
    # Convertir en array numpy de type float et supprimer les valeurs NaN
    points = df.astype(float).to_numpy()
    mask = ~np.isnan(points).any(axis=1)
    points = points[mask]

    if len(points) == 0:
        print("Aucun point trouvé")
        return

    # Pour chaque point qui a un y négatif, on inverse x et y
    for i in range(len(points)):
        if points[i, 1] < 0:  # Si y est négatif
            points[i, 0] = -points[i, 0]  # Inverse x
            points[i, 1] = -points[i, 1]  # Inverse y

    # Normaliser les données
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # Calculer le score de silhouette pour différentes valeurs de k
    k_range = range(
        2, 7
    )  # Réduire la plage pour se concentrer sur les valeurs pertinentes
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(points_scaled)
        # Ajouter une pénalité pour les clusters trop éloignés de 4
        score = silhouette_score(points_scaled, clusters)
        silhouette_scores.append(score)

    # Tracer la courbe des scores de silhouette
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, "bx-")
    plt.xlabel("k (nombre de clusters)")
    plt.ylabel("Score de silhouette moyen")
    plt.title(f"Scores de silhouette pour {name}")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Trouver le nombre optimal de clusters (maximum du score de silhouette)
    optimal_k = k_range[np.argmax(silhouette_scores)]

    # Marquer le point optimal
    plt.plot(
        optimal_k,
        silhouette_scores[optimal_k - 2],  # -2 car k_range commence à 2
        "ro",
        markersize=10,
        label=f"Optimal (k={optimal_k})",
    )
    plt.legend()

    # Sauvegarder la figure
    plt.savefig(
        f"images/silhouette_scores_{name.replace(' ', '_').lower()}.svg"
    )
    plt.close()

    print(f"\nAnalyse pour {name}:")
    print(f"Nombre optimal de clusters suggéré : {optimal_k}")
    print(f"Score de silhouette optimal : {max(silhouette_scores):.3f}")
    print(f"Nombre total de points : {len(points)}")

    # Réaliser le clustering avec le nombre optimal de clusters
    optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    clusters = optimal_kmeans.fit_predict(points_scaled)

    # Visualiser les clusters sur un scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        points[:, 0], points[:, 1], c=clusters, cmap="viridis"
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.title(f"Distribution des points en {optimal_k} clusters")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Sauvegarder la figure de distribution
    plt.savefig(
        f"images/silhouette_distribution_{name.replace(' ', '_').lower()}.svg"
    )
    plt.close()

    # Calculer et afficher les statistiques détaillées des clusters
    silhouette_values = silhouette_samples(points_scaled, clusters)
    print("\nStatistiques des clusters:")
    for i in range(optimal_k):
        cluster_points = points[clusters == i]
        cluster_silhouette = silhouette_values[clusters == i]

        print(f"\nCluster {i+1}:")
        print(f"Nombre de points : {len(cluster_points)}")
        print(
            f"Centre : ({np.mean(cluster_points[:, 0]):.2f}, {np.mean(cluster_points[:, 1]):.2f})"
        )
        print(
            f"Écart-type : ({np.std(cluster_points[:, 0]):.2f}, {np.std(cluster_points[:, 1]):.2f})"
        )
        print(f"Score de silhouette moyen : {np.mean(cluster_silhouette):.3f}")


def main():
    """Fonction principale."""
    # Connexion à la base de données
    DB_PATH = "BDD_avec_cluster.db"
    conn = sqlite3.connect(DB_PATH)

    # Récupération de tous les retours de service
    query = """
    SELECT coor_balle_x, coor_balle_y
    FROM Liste_des_coups
    WHERE coor_balle_x IS NOT NULL 
    AND coor_balle_y IS NOT NULL
    AND num_coup = 2  -- Seulement les retours de service
    """

    # Charger les données et analyser
    df = pd.read_sql_query(query, conn)
    analyze_silhouette("Tous les retours de service", df)

    conn.close()


if __name__ == "__main__":
    main()
