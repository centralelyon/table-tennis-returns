import sqlite3

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans


def place_on_side(
    positions_serveur: list[tuple[float, float]] = [],
    positions_rebonds_serveur: list[tuple[float, float]] = [],
    positions_adv: list[tuple[float, float]] = [],
    positions_rebonds_adv: list[tuple[float, float]] = [],
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    """
    Place les points du bon côté de la table (en haut pour les seconds rebonds de service, et en bas pour les retours).
    Ne modifie pas les points déjà du bon côté de la table.

    Args:
        positions_serveur: Liste de couples de coordonnées du serveur.
        positions_rebonds_serveur: Liste de couples de coordonnées du second rebond de service.
        positions_adv: Liste de couples de coordonnées du retourneur.
        positions_rebonds_serveur: Liste de couples de coordonnées du retour.

    Returns:
        tuple: Les quatre listes de coordonnées modifiées.
    """
    if positions_serveur:
        for i in range(len(positions_serveur)):
            if positions_serveur[i] is not None:
                if (
                    positions_serveur[i][1] > 0
                ):  # On vérifie le côté de la table
                    positions_serveur[i] = (
                        -positions_serveur[i][0],
                        -positions_serveur[i][1],
                    )

    if positions_rebonds_serveur:
        for i in range(len(positions_rebonds_serveur)):
            if positions_rebonds_serveur[i] is not None:
                if (
                    positions_rebonds_serveur[i][1] < 0
                ):  # On vérifie le côté de la table
                    positions_rebonds_serveur[i] = (
                        -positions_rebonds_serveur[i][0],
                        -positions_rebonds_serveur[i][1],
                    )

    if positions_adv:
        for i in range(len(positions_adv)):
            if positions_adv[i] is not None:
                if positions_adv[i][1] < 0:  # On vérifie le côté de la table
                    positions_adv[i] = (
                        -positions_adv[i][0],
                        -positions_adv[i][1],
                    )

    if positions_rebonds_adv:
        for i in range(len(positions_rebonds_adv)):
            if (
                positions_rebonds_adv[i] is not None
                and positions_rebonds_adv[i][0] is not None
                and positions_rebonds_adv[i][1] is not None
            ):
                if (
                    positions_rebonds_adv[i][1] > 0
                ):  # On vérifie le côté de la table
                    positions_rebonds_adv[i] = (
                        -positions_rebonds_adv[i][0],
                        -positions_rebonds_adv[i][1],
                    )

    return (
        positions_serveur,
        positions_rebonds_serveur,
        positions_adv,
        positions_rebonds_adv,
    )


def k_moyennes(
    coordonnees: list[tuple[float, float]], k: int, random_state: int = 42
) -> tuple[list[int], list[tuple[float, float]], list[ConvexHull | None]]:
    """
    Applique l'algorithme des k-moyennes.

    Args:
        coordonnees: Liste des points sur lesquels appliquer les k-moyennes.
        k: Nombre de clusters.
        random_state: Graine fixée pour reproductibilité.

    Returns:
        tuple: (labels, barycentres, enveloppes)
            labels: Liste des labels.
            barycentres: Liste des coordonnées des barycentres.
            enveloppes: Liste des enveloppes des clusters.
    """
    # Formatage des données
    X = np.array(coordonnees)

    if X.shape[0] < k:  # Pas assez de points
        return [], [], []

    # On applique l'alogrithme des k moyennes
    kmeans = KMeans(n_clusters=k, random_state=random_state)

    kmeans.fit(X)

    # Récupération des labels des différents points ainsi que des barycentres des clusters
    labels = kmeans.labels_
    barycentres = kmeans.cluster_centers_

    # Calcul des enveloppes pour chaque cluster
    enveloppes = []
    for i in range(k):
        # Points appartenant au cluster i
        points_du_cluster = X[labels == i]

        # On vérifie qu'on a assez de points pour former l'enveloppe et que les points ne sont pas colinéaires
        if len(points_du_cluster) >= 3:
            try:
                hull = ConvexHull(
                    points_du_cluster, qhull_options="QJ"
                )  # Utiliser QJ pour "jogger" les points
                enveloppes.append(hull)
            except QhullError:
                # Si une erreur de colinéarité survient, on ne crée pas l'enveloppe
                enveloppes.append(None)
        else:
            enveloppes.append(None)

    return labels, barycentres, enveloppes


def get_clusters_all_sets(
    positions_rebonds_serveur: list[tuple[float, float]] = [],
    positions_rebonds_adv: list[tuple[float, float]] = [],
    k_serveur: int = 4,
    k_adv: int = 4,
    random_state: int = 42,
) -> tuple[
    list[int], list[ConvexHull | None], list[int], list[ConvexHull | None]
]:
    """
    Récupère les clusters pour les rebonds données en arguments.

    Args:
        positions_rebonds_serveur: Liste des positions de rebonds de service.
        positions_rebonds_adv: Liste des positions de rebonds de retourneur.
        k_serveur: Nombre de clusters pour les services.
        k_adv: Nombre de clusters pour les retours.

    Returns:
        tuple: (labels_serveur, enveloppes_serveur, labels_adv, enveloppes_adv)
            labels_serveur: Liste des labels des services.
            enveloppes_serveur: Liste des enveloppes pour le service.
            labels_adv: Liste des labels pour les retours.
            enveloppes_adv: Liste des enveloppes pour les retours.
    """
    _, positions_rebonds_serveur, _, positions_rebonds_adv = place_on_side(
        [], positions_rebonds_serveur, [], positions_rebonds_adv
    )

    labels_serveur, labels_adv, enveloppes_serveur, enveloppes_adv = (
        [],
        [],
        [],
        [],
    )

    if positions_rebonds_serveur:
        labels_serveur, _, enveloppes_serveur = k_moyennes(
            positions_rebonds_serveur, k_serveur, random_state
        )

    if positions_rebonds_adv:
        labels_adv, _, enveloppes_adv = k_moyennes(
            positions_rebonds_adv, k_adv, random_state
        )

    return labels_serveur, enveloppes_serveur, labels_adv, enveloppes_adv


def get_colors_win_lose(
    points: list[tuple[float, float]],
) -> list[tuple[str, str]]:
    """
    Renvoie la liste des gagnants des points correspondants à une liste de coordonnées donnée,
    dans le but de colorer les points selon le vainqueur sur la figure.

    Utilise une comparaison approximative pour trouver les points dans la base de données,
    indépendamment du signe et avec une précision réduite.

    Args:
        points: Liste des coordonnées des points à étudier.

    Returns:
        list: Liste des tuples (gagnant, rôle) pour les points correspondants,
              où rôle est "server" ou "returner".
    """

    gagnants = []
    # Définir la précision pour la comparaison
    precision = 0.0001

    conn = sqlite3.connect("BDD_avec_cluster.db")
    try:
        for point in points:
            x, y = point[0], point[1]
            # Utiliser la valeur absolue et arrondir pour la comparaison
            query = """
                SELECT *
                FROM Liste_des_coups
                WHERE ABS(coor_balle_x - ?) < ? AND ABS(coor_balle_y - ?) < ?
                LIMIT 1
            """
            result = pd.read_sql_query(
                query, conn, params=(x, precision, y, precision)
            )

            if result.empty:
                gagnants.append(None)
            else:
                winner = result["winner"].iloc[0]
                serveur = result["serveur"].iloc[0]
                if winner == serveur:
                    gagnants.append((winner, "server"))
                else:
                    gagnants.append((winner, "returner"))
    finally:
        conn.close()

    return gagnants
