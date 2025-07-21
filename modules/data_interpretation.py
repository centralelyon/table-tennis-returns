import math
import os
import sqlite3
import sys

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib import rc, rcParams
from matplotlib.ticker import MaxNLocator
from scipy.stats import chi2_contingency

from modules.data_processing import get_clusters_all_sets, place_on_side
from modules.figure_module import Figure

chemin_dossier_parent = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(chemin_dossier_parent)
chemin_bdd = os.path.join(chemin_dossier_parent, "BDD_avec_cluster.db")


def recup_match(nom_joueur):
    """
    Retourne la liste des matchs dans lesquels apparaît le joueur donné.

    Args:
        nom_joueur (str): Nom du joueur à rechercher.

    Returns:
        list: Liste des noms de matchs où le joueur apparaît.
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete = f"""
            SELECT distinct Gamename
            FROM Liste_des_coups
            WHERE (joueur_sur = '{nom_joueur}' OR  joueur_frappe= '{nom_joueur}')
        """
        df = pd.read_sql_query(requete, conn)
    finally:
        conn.close()
    liste_match = [df["Gamename"][i] for i in range(len(df["Gamename"]))]
    return liste_match


def recup_service(nom_joueur: str, match: str):
    """
    Retourne la liste des positions de service du joueur pour un match donné.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste de tuples (x, y) des positions de service.
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete_service = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {1}
            AND Gamename = '{match}'
            AND joueur_frappe= '{nom_joueur}'
        """
        df_service = pd.read_sql_query(requete_service, conn)
    finally:
        conn.close()
    positions_services = [
        (df_service["coor_balle_x"][i], df_service["coor_balle_y"][i])
        for i in range(len(df_service["coor_balle_x"]))
        if not math.isnan(df_service["coor_balle_x"][i])
    ]
    return positions_services


def recup_retours(nom_joueur: str, match: str):
    """
    Retourne les positions des retours et des coups précédents pour un joueur au retour dans un match.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        tuple: (positions coups précédents, positions retours adverses)
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete_rebonds_adv = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {2}
            AND Gamename = '{match}'
            AND num_point IS NOT NULL
            AND joueur_frappe= '{nom_joueur}'
        """
        df_rebonds_adv = pd.read_sql_query(requete_rebonds_adv, conn)
    finally:
        conn.close()
    positions_rebonds_prece = [
        (
            df_rebonds_adv["pos_balle_x_prece"][i],
            df_rebonds_adv["pos_balle_y_prece"][i],
        )
        for i in range(len(df_rebonds_adv["pos_balle_x_prece"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    positions_rebonds_adv = [
        (df_rebonds_adv["coor_balle_x"][i], df_rebonds_adv["coor_balle_y"][i])
        for i in range(len(df_rebonds_adv["coor_balle_x"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    return positions_rebonds_prece, positions_rebonds_adv


def recup_retours_si_3emecoup(nom_joueur: str, match: str):
    """
    Retourne les positions des retours et des coups précédents pour un joueur au retour, uniquement si un 3ème coup est joué.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        tuple: (positions coups précédents, positions retours adverses)
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete_rebonds_adv = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {2}
            AND Gamename = '{match}'
            AND num_point IS NOT NULL
            AND joueur_frappe= '{nom_joueur}'
            AND nb_coup >= {3}
        """
        df_rebonds_adv = pd.read_sql_query(requete_rebonds_adv, conn)
    finally:
        conn.close()
    positions_rebonds_prece = [
        (
            df_rebonds_adv["pos_balle_x_prece"][i],
            df_rebonds_adv["pos_balle_y_prece"][i],
        )
        for i in range(len(df_rebonds_adv["pos_balle_x_prece"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    positions_rebonds_adv = [
        (df_rebonds_adv["coor_balle_x"][i], df_rebonds_adv["coor_balle_y"][i])
        for i in range(len(df_rebonds_adv["coor_balle_x"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    return positions_rebonds_prece, positions_rebonds_adv


def recup_donnees(nom_joueur: str, match: str):
    """
    Retourne différentes listes d'informations sur les points d'un match pour un joueur donné.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        tuple: (vainqueurs points précédents, vainqueurs actuels, effets, latéralité, numéros de set)
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete_rebonds_adv = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {2}
            AND Gamename = '{match}'
            AND num_point IS NOT NULL
            AND joueur_frappe= '{nom_joueur}'
        """
        df_rebonds_adv = pd.read_sql_query(requete_rebonds_adv, conn)
    finally:
        conn.close()
    winner = [
        df_rebonds_adv["winner"][i]
        for i in range(len(df_rebonds_adv["coor_balle_x"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    winner_prec = [winner[i - 1] for i in range(1, len(winner))]
    vainqueurs = winner
    liste_effets = [
        df_rebonds_adv["effet_coup"][i]
        for i in range(len(df_rebonds_adv["coor_balle_x"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    liste_lateralite_retour = [
        df_rebonds_adv["lateralite"][i]
        for i in range(len(df_rebonds_adv["coor_balle_x"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    liste_numero_set = [
        df_rebonds_adv["numero_set"][i]
        for i in range(len(df_rebonds_adv["coor_balle_x"]))
        if not math.isnan(df_rebonds_adv["coor_balle_x"][i])
    ]
    return (
        winner_prec,
        vainqueurs,
        liste_effets,
        liste_lateralite_retour,
        liste_numero_set,
    )


def recup_troisieme_coup(nom_joueur: str, match: str):
    """
    Récupère une liste où le i-ème élément de la liste vaut True si le 3ème coup est un coup gagnant pour l'adversaire.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste de booléens indiquant si le 3ème coup est un coup gagnant pour l'adversaire.
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete_troisieme_coup = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {3}
            AND Gamename = '{match}'
            AND num_point IS NOT NULL
            AND joueur_sur= '{nom_joueur}'
        """
        df_troisieme_coup = pd.read_sql_query(requete_troisieme_coup, conn)
    finally:
        conn.close()
    liste_bool = []
    for i in range(len(df_troisieme_coup)):
        if df_troisieme_coup["winner"][i] != nom_joueur and df_troisieme_coup[
            "nb_coup"
        ][i] in {3, 4}:
            liste_bool.append(True)
        else:
            liste_bool.append(False)

    return liste_bool


def table_contingence(data_dict):
    """
    Construit une table de contingence à partir d'un dictionnaire de données.

    Args:
        data_dict (dict): Dictionnaire contenant les données à croiser.

    Returns:
        pd.DataFrame: Table de contingence.
    """
    empty_checker = False
    for value in data_dict.values():
        if list(value) == []:
            empty_checker = True
            break
    if empty_checker:
        return
    data = pd.DataFrame(data_dict)
    tableau_contingence = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
    return tableau_contingence


def test_chi2(data_dict: dict):
    """
    Effectue un test du chi2 sur une table de contingence construite à partir d'un dictionnaire de données.

    Args:
        data_dict (dict): Dictionnaire contenant les données à croiser.

    Returns:
        tuple: (valeur du chi2, p-value, degrés de liberté)
    """
    empty_checker = False
    for value in data_dict.values():
        if list(value) == []:
            empty_checker = True
            break
    if empty_checker:
        return
    tableau_contingence = table_contingence(data_dict)
    chi2_value, p_value, ddl, _ = chi2_contingency(tableau_contingence)
    return chi2_value, p_value, ddl


def test_chi2_serv(label_services, label_retours):
    """
    Effectue un test du chi2 entre les labels de service et de retour.

    Args:
        label_services (list): Labels des services.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Service": label_services, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_lateralite(liste_lateralite, label_retours):
    """
    Effectue un test du chi2 entre la latéralité et les labels de retour.

    Args:
        liste_lateralite (list): Liste des latéralités.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Latéralité": liste_lateralite, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_score(Gagnants, label_retours):
    """
    Effectue un test du chi2 entre les gagnants des points et les labels de retour.

    Args:
        Gagnants (list): Liste des gagnants des points.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Gagnant": Gagnants, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_win(Vainqueurs, label_retours):
    """
    Effectue un test du chi2 entre les vainqueurs et les labels de retour.

    Args:
        Vainqueurs (list): Liste des vainqueurs.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Vainqueur": Vainqueurs, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_effet(liste_effets, label_retours):
    """
    Effectue un test du chi2 entre les effets et les labels de retour.

    Args:
        liste_effets (list): Liste des effets.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Effet": liste_effets, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_domination(liste_domination, label_retours):
    """
    Effectue un test du chi2 entre la domination et les labels de retour.

    Args:
        liste_domination (list): Liste des dominations.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Dominant": liste_domination, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_effet_vainq(Vainqueurs, liste_effets):
    """
    Effectue un test du chi2 entre les vainqueurs et les effets.

    Args:
        Vainqueurs (list): Liste des vainqueurs.
        liste_effets (list): Liste des effets.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Vainqueur": Vainqueurs, "Retour": liste_effets}
    return test_chi2(data)


def test_chi2_pression(liste_pression, label_retours):
    """
    Effectue un test du chi2 entre la pression et les labels de retour.

    Args:
        liste_pression (list): Liste des pressions.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Pression": liste_pression, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_set(liste_set, label_retours):
    """
    Effectue un test du chi2 entre le numéro de set et les labels de retour.

    Args:
        liste_set (list): Liste des numéros de set.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {"Numéro set": liste_set, "Retour": label_retours}
    return test_chi2(data)


def test_chi2_3eme_coup(liste_3eme_coup, label_retours):
    """
    Effectue un test du chi2 entre le 3ème coup menant à un point gagné pour l'adversaire et les labels de retour.

    Args:
        liste_3eme_coup (list): Liste des booléens pour le 3ème coup.
        label_retours (list): Labels des retours.

    Returns:
        tuple: Résultat du test du chi2.
    """
    data = {
        "3eme coup menant à un point gagné pour adversaire": liste_3eme_coup,
        "Retour": label_retours,
    }
    return test_chi2(data)


def get_data_from_table(
    rows: list, columns: list, rows_data, columns_data
) -> list[("Figure", tuple)]:
    """
    Récupère et organise les données pour chaque case d'une table de contingence, puis génère une figure pour chaque case.

    Args:
        rows (list): Liste des valeurs pour les lignes de la table.
        columns (list): Liste des valeurs pour les colonnes de la table.
        rows_data (list): Données associées aux lignes (ex : coordonnées).
        columns_data (list): Données associées aux colonnes (ex : coordonnées).

    Returns:
        list: Liste de tuples (Figure, (row, column)) pour chaque case de la table.
    """
    cote_coup = [
        2 for _ in range(len(columns_data))
    ]  # A modifier pour généraliser et placer des points des deux côtés de la table
    fig_list = []

    _, _, _, columns_data = place_on_side([], [], [], columns_data)

    # Si c'est des coordonnées.
    if rows_data and isinstance(rows_data[0], tuple):
        _, rows_data, _, _ = place_on_side([], rows_data, [], [])

    points = {}

    for i in range(len(rows)):
        row_key = str(rows[i])
        column_key = str(columns[i])

        # Initialiser le sous-dictionnaire si la clé n'existe pas
        if row_key not in points:
            points[row_key] = {}

        # Initialiser la liste si la clé n'existe pas
        if column_key not in points[row_key]:
            points[row_key][column_key] = []

        # Ajouter les données
        points[row_key][column_key].append(columns_data[i])

    for row in set(rows):
        for column in set(columns):

            if (
                str(row) in points and str(column) in points[str(row)]
            ):  # Traite le cas où la cellule vaut 0

                fig = Figure(False)
                fig.add_positions_rebonds(
                    [points[str(row)][str(column)]], cote_coup
                )

                fig_list.append((fig, (row, column)))

    return fig_list


def get_data_from_table_old(
    rows: list,
    columns: list,
    rows_id: list[tuple],
    columns_id: list[tuple],
    rows_label: str,
    columns_label: str,
) -> list[("Figure", tuple)]:
    """
    Récupère et affiche les points correspondants à une case d'une table de contingence.

    Args:
        rows (list): Données pour chaque ligne de la table de contingence.
        columns (list): Données pour chaque colonne de la table de contingence.
        rows_id (list of tuple): Couples (id_match, id_coup) pour chaque élément de rows.
        columns_id (list of tuple): Couples (id_match, id_coup) pour chaque élément de columns.
        rows_label (str): Nom des lignes.
        columns_label (str): Nom des colonnes.

    Returns:
        list: Liste de tuples (Figure, (row, column)) pour chaque case de la table.
    """
    con = sqlite3.connect("BDD_avec_cluster.db")
    data_lateralite_retour = {rows_label: rows, columns_label: columns}

    df_lateralite_retour = pd.DataFrame(
        {
            rows_label: rows,
            columns_label: columns,
            rows_label + "_id": rows_id,
            columns_label + "_id": columns_id,
        }
    )

    table_cont = table_contingence(data_lateralite_retour)

    print(table_cont, "\n")

    ids = (
        []
    )  # Contient les couples (id_match, id_coup) pour chaque case du tableau
    ids_case = (
        []
    )  # Contient les coordonnées de chaque case (abscisse et ordonnée dans la table de contingence)

    for row in df_lateralite_retour[rows_label].unique():
        for column in df_lateralite_retour[columns_label].unique():
            # Filtrer les éléments correspondants dans les listes d'ID
            elements = df_lateralite_retour[
                (df_lateralite_retour[rows_label] == row)
                & (df_lateralite_retour[columns_label] == column)
            ]
            if not elements.empty:
                ids.append(elements[columns_label + "_id"].values)
                ids_case.append((row, column))

    points = (
        []
    )  # Récupère les coordonnées des points à partir des id (id_match, id_coup)
    cote_coup = (
        []
    )  # Contient les num_coup pour placer les points du bon côté sur la figure

    for i in range(len(ids)):
        point = []
        for match, coup in ids[i]:
            requete_lateralite_id = f"SELECT * FROM Liste_des_coups WHERE IdMatch = {match} AND IdCoup = {coup}"
            df_lateralite_id = pd.read_sql_query(requete_lateralite_id, con)
            point.append(
                (
                    float(df_lateralite_id["coor_balle_x"]),
                    float(df_lateralite_id["coor_balle_y"]),
                )
            )
        cote_coup.append(df_lateralite_id["num_coup"][0])
        points.append(point)

    fig_list = (
        []
    )  # Contient la liste des couples (figures, coord) pour chaque case (où coord est la coordonnée dans la table de contingence).
    for num_case in range(len(points)):
        if cote_coup[num_case] % 2 == 1:
            _, points[num_case], _, _ = place_on_side(
                [], points[num_case], [], []
            )
        else:
            _, _, _, points[num_case] = place_on_side(
                [], [], [], points[num_case]
            )

        fig = Figure(False)
        fig.add_positions_rebonds([points[num_case]], cote_coup)

        fig_list.append((fig, ids_case[num_case]))

    return fig_list


def get_data_chi2_serves_returns(
    figure_list: list["Figure"],
    num_coups: list,
    indice: int,
    joueur: str,
    effet_list: list = [],
    random_state: int = 42,
):
    """
    Récupère et met en forme les données nécessaires pour le test du chi2 entre les zones de service et de retour.

    Args:
        figure_list (list): Liste des objets Figure.
        num_coups (list): Liste des numéros de coups.
        indice (int): Indice du joueur à traiter.
        joueur (str): Nom du joueur.
        effet_list (list, optionnel): Liste des effets à prendre en compte.
        random_state (int, optionnel): Graine aléatoire pour le clustering.

    Returns:
        tuple: Données formatées pour le test du chi2 (voir code pour le détail).
    """
    match_list = (
        Figure().get_list_match()
    )  # Récupère la liste de tous les matchs de la BDD.

    # Ajout des rebonds
    rebonds_list = []
    precedent_list = []
    rebonds_id_list = []
    precedent_id_list = []
    for match in match_list:
        if effet_list:
            rebonds = figure_list[indice].get_rebonds(
                joueur, num_coups[indice], match, effet_list[indice]
            )
        else:
            rebonds = figure_list[indice].get_rebonds(
                joueur, num_coups[indice], match
            )
        rebonds_list.append(rebonds[0])
        precedent_list.append(rebonds[1])
        rebonds_id_list.append(rebonds[2])
        precedent_id_list.append(rebonds[3])

    for i in range(len(rebonds_list)):
        # On place les points du bon côté de la table.
        if num_coups[0][indice] % 2 == 0:
            _, precedent_list[i], _, rebonds_list[i] = place_on_side(
                [], precedent_list[i], [], rebonds_list[i]
            )
        if num_coups[0][indice] % 2 == 1:
            _, rebonds_list[i], _, precedent_list[i] = place_on_side(
                [], rebonds_list[i], [], precedent_list[i]
            )

    # Nettoyage des données (suppression des None et (nan, nan))
    (
        filtered_data_1,
        filtered_data_2,
        filtered_data_1_id,
        filtered_data_2_id,
    ) = (
        [],
        [],
        [],
        [],
    )
    for sublist1, sublist2, sublist3, sublist4 in zip(
        rebonds_list, precedent_list, rebonds_id_list, precedent_id_list
    ):
        temp1, temp2, temp3, temp4 = [], [], [], []
        for coords1, coords2, id1, id2 in zip(
            sublist1, sublist2, sublist3, sublist4
        ):
            if (
                coords1 is not None
                and coords1[0] is not None
                and coords1[1] is not None
                and not (math.isnan(coords1[0]) or math.isnan(coords1[1]))
            ):
                temp1.append(coords1)
                temp2.append(coords2)
                temp3.append(id1)
                temp4.append(id2)
        filtered_data_1.append(temp1)
        filtered_data_2.append(temp2)
        filtered_data_1_id.append(temp3)
        filtered_data_2_id.append(temp4)

    rebonds_list = filtered_data_1
    precedent_list = filtered_data_2
    rebonds_id_list = filtered_data_1_id
    precedent_id_list = filtered_data_2_id

    flattened_rebonds_list = [
        item
        for sublist in rebonds_list
        for item in sublist
        if item is not None and not np.isnan(item[0])
    ]
    flattened_precedent_list = [
        item
        for sublist in precedent_list
        for item in sublist
        if item is not None and not np.isnan(item[0])
    ]
    flattened_rebonds_id_list = [
        item
        for sublist in rebonds_id_list
        for item in sublist
        if item is not None and not np.isnan(item[0])
    ]
    flattened_precedent_id_list = [
        item
        for sublist in precedent_id_list
        for item in sublist
        if item is not None and not np.isnan(item[0])
    ]

    # Ajouts des rebonds à la figure
    figure_list[indice].add_positions_rebonds(
        [flattened_rebonds_list],
        [num_coups[0][indice] % 2],
    )
    figure_list[indice].add_positions_rebonds(
        [flattened_precedent_list],
        [1 - num_coups[0][indice] % 2],
    )

    labels_retourneur, labels_serveur = figure_list[indice].add_clusters(
        [flattened_rebonds_list, flattened_precedent_list],
        [4, 4],
        [num_coups[0][indice] % 2, 1 - num_coups[0][indice] % 2],
        random_state,
    )

    return (
        figure_list,
        labels_serveur,
        labels_retourneur,
        flattened_precedent_id_list,
        flattened_rebonds_id_list,
        flattened_precedent_list,
        flattened_rebonds_list,
    )


def get_data_chi2_winners_returns(
    figure_list: list["Figure"],
    num_coups: list,
    indice: int,
    joueur: str,
    effet_list: list = [],
    random_state: int = 42,
):
    """
    Récupère et met en forme les données nécessaires pour le test du chi2 entre le vainqueur du point actuel et les zones de retour.

    Args:
        figure_list (list): Liste des objets Figure.
        num_coups (list): Liste des numéros de coups.
        indice (int): Indice du joueur à traiter.
        joueur (str): Nom du joueur.
        effet_list (list, optionnel): Liste des effets à prendre en compte.
        random_state (int, optionnel): Graine aléatoire pour le clustering.

    Returns:
        tuple: Données formatées pour le test du chi2 (voir code pour le détail).
    """
    match_list = (
        Figure().get_list_match()
    )  # Récupère la liste de tous les matchs de la BDD.

    # Ajout des rebonds
    rebonds_list = []
    rebonds_id_list = []
    for match in match_list:
        if effet_list:
            rebonds = figure_list[indice].get_rebonds(
                joueur, num_coups[indice], match, effet_list[indice]
            )
        else:
            rebonds = figure_list[indice].get_rebonds(
                joueur, num_coups[indice], match
            )
        rebonds_list.append(rebonds[0])
        rebonds_id_list.append(rebonds[2])

    for i in range(len(rebonds_list)):
        # On place les points du bon côté de la table.
        if num_coups[0][indice] % 2 == 0:
            _, _, _, rebonds_list[i] = place_on_side(
                [], [], [], rebonds_list[i]
            )
        if num_coups[0][indice] % 2 == 1:
            _, rebonds_list[i], _, _ = place_on_side(
                [], rebonds_list[i], [], []
            )

    # Ajouts des rebonds à la figure
    figure_list[indice].add_positions_rebonds(
        rebonds_list,
        [num_coups[0][indice] % 2 for _ in range(len(rebonds_list))],
    )

    # Nettoyage des données (suppression des None et (nan, nan))
    filtered_data_1, filtered_data_1_id = [], []
    for sublist1, sublist2 in zip(rebonds_list, rebonds_id_list):
        temp1, temp2 = [], []
        for coords1, id1 in zip(sublist1, sublist2):
            if (
                coords1 is not None
                and coords1[0] is not None
                and coords1[1] is not None
                and not (math.isnan(coords1[0]) or math.isnan(coords1[1]))
            ):
                temp1.append(coords1)
                temp2.append(id1)
        filtered_data_1.append(temp1)
        filtered_data_1_id.append(temp2)

    rebonds_list = filtered_data_1
    rebonds_id_list = filtered_data_1_id

    flattened_rebonds_list = [
        item
        for sublist in rebonds_list
        for item in sublist
        if item is not None and not np.isnan(item[0])
    ]
    flattened_rebonds_id_list = [
        item
        for sublist in rebonds_id_list
        for item in sublist
        if item is not None and not np.isnan(item[0])
    ]

    figure_list[indice].add_positions_rebonds([flattened_rebonds_list], [2])

    labels_retourneur = figure_list[indice].add_clusters(
        [flattened_rebonds_list], [4], [2], random_state
    )[0]

    vainqueurs = []
    con = sqlite3.connect(chemin_bdd)

    for i in range(len(flattened_rebonds_list)):
        requete_vainqueur = f"SELECT * FROM Liste_des_coups WHERE IdMatch = {flattened_rebonds_id_list[i][0]} AND IdCoup = {flattened_rebonds_id_list[i][1]}"
        df_vainqueur = pd.read_sql_query(requete_vainqueur, con)
        winner = [
            df_vainqueur["winner"][i]
            for i in range(len(df_vainqueur["coor_balle_x"]))
        ]

        vainqueurs.append(winner)

    flattened_vainqueurs = [
        item for sublist in vainqueurs for item in sublist if item is not None
    ]

    # Format : figure_list, key1 table contingence, key2 table contingence, id1, id2, data1, data2
    return (
        figure_list,
        flattened_vainqueurs,
        labels_retourneur,
        [],
        flattened_rebonds_id_list,
        [],
        flattened_rebonds_list,
    )


def truncate_float(value, decimals):
    """
    Tronque un nombre flottant à un nombre de décimales donné (sans arrondi).

    Args:
        value (float): Valeur à tronquer.
        decimals (int): Nombre de décimales à conserver.

    Returns:
        float: Valeur tronquée.
    """
    factor = 10**decimals
    return int(value * factor) / factor


def get_data_chi2_domination_returns(
    figure_list: list["Figure"],
    indice: int,
    joueur: str,
    random_state: int = 42,
):
    """
    Récupère et met en forme les données nécessaires pour le test du chi2 entre la domination et les zones de retour.

    Args:
        figure_list (list): Liste des objets Figure.
        indice (int): Indice du joueur à traiter.
        joueur (str): Nom du joueur.
        random_state (int, optionnel): Graine aléatoire pour le clustering.

    Returns:
        tuple: Données formatées pour le test du chi2 (voir code pour le détail).
    """
    con = sqlite3.connect(chemin_bdd)
    match_list_player = Figure().get_list_match(joueur)
    dominations = []
    flattened_rebonds_list = []  # Liste des coordonnées de

    for i in range(len(match_list_player)):
        _, domination, _, coord = data_domination(joueur, match_list_player[i])
        dominations += domination
        flattened_rebonds_list += coord

    filtered_rebonds = []
    filtered_dominations = []
    for rebond, domination in zip(flattened_rebonds_list, dominations):
        if not math.isnan(
            rebond[0]
        ):  # Garde uniquement si rebond[0] n'est pas NaN
            filtered_rebonds.append(rebond)
            filtered_dominations.append(domination)

    flattened_rebonds_list = filtered_rebonds
    dominations = filtered_dominations

    figure_list[indice].add_positions_rebonds([flattened_rebonds_list], [2])

    labels_retourneur = figure_list[indice].add_clusters(
        [flattened_rebonds_list], [4], [2], random_state
    )[0]
    flattened_rebonds_id_list = []

    for i in range(len(flattened_rebonds_list)):
        requete_ids = (  # Bug dans SQL/BDD, quand il y a trop de chiffre, la donnée n'est pas reconnue : on tronque
            f"SELECT * FROM Liste_des_coups "
            f"WHERE CAST(coor_balle_x AS TEXT) LIKE '{truncate_float(flattened_rebonds_list[i][0], 9)}%' "
            f"AND CAST(coor_balle_y AS TEXT) LIKE '{truncate_float(flattened_rebonds_list[i][1], 9)}%'"
        )

        df_ids = pd.read_sql_query(requete_ids, con)
        if list(df_ids["IdMatch"]) != []:
            if len(df_ids["IdMatch"] == 1):
                flattened_rebonds_id_list.append(
                    (list(df_ids["IdMatch"])[0], list(df_ids["IdCoup"])[0])
                )
            else:
                flattened_rebonds_id_list.append(
                    (list(df_ids["IdMatch"])[0], list(df_ids["IdCoup"])[0])
                )

    # Format : figure_list, key1 table contingence, key2 table contingence, id1, id2, data1, data2
    return (
        figure_list,
        dominations,
        labels_retourneur,
        [],
        flattened_rebonds_id_list,
        [],
        flattened_rebonds_list,
    )


def get_data_chi2_pression_returns(
    figure_list: list["Figure"],
    indice: int,
    joueur: str,
    random_state: int = 42,
):
    """
    Récupère et met en forme les données nécessaires pour le test du chi2 entre la pression et les zones de retour.

    Args:
        figure_list (list): Liste des objets Figure.
        indice (int): Indice du joueur à traiter.
        joueur (str): Nom du joueur.
        random_state (int, optionnel): Graine aléatoire pour le clustering.

    Returns:
        tuple: Données formatées pour le test du chi2 (voir code pour le détail).
    """
    con = sqlite3.connect(chemin_bdd)
    match_list_player = Figure().get_list_match(joueur)
    pressions = []
    flattened_rebonds_list = []  # Liste des coordonnées de

    for i in range(len(match_list_player)):
        pression, coord = recup_pression_si_retour(
            joueur, match_list_player[i]
        )
        pressions += pression
        flattened_rebonds_list += coord

    filtered_rebonds = []
    filtered_pressions = []
    for rebond, pression in zip(flattened_rebonds_list, pressions):
        if not math.isnan(
            rebond[0]
        ):  # Garde uniquement si rebond[0] n'est pas NaN
            filtered_rebonds.append(rebond)
            filtered_pressions.append(pression)

    flattened_rebonds_list = filtered_rebonds
    pressions = filtered_pressions

    figure_list[indice].add_positions_rebonds([flattened_rebonds_list], [2])

    labels_retourneur = figure_list[indice].add_clusters(
        [flattened_rebonds_list], [4], [2], random_state
    )[0]
    flattened_rebonds_id_list = []

    for i in range(len(flattened_rebonds_list)):
        requete_ids = (  # Bug dans SQL/BDD, quand il y a trop de chiffre, la donnée n'est pas reconnue : on tronque
            f"SELECT * FROM Liste_des_coups "
            f"WHERE CAST(coor_balle_x AS TEXT) LIKE '{truncate_float(flattened_rebonds_list[i][0], 9)}%' "
            f"AND CAST(coor_balle_y AS TEXT) LIKE '{truncate_float(flattened_rebonds_list[i][1], 9)}%'"
        )

        df_ids = pd.read_sql_query(requete_ids, con)
        if list(df_ids["IdMatch"]) != []:
            if len(df_ids["IdMatch"] == 1):
                flattened_rebonds_id_list.append(
                    (list(df_ids["IdMatch"])[0], list(df_ids["IdCoup"])[0])
                )
            else:
                flattened_rebonds_id_list.append(
                    (list(df_ids["IdMatch"])[0], list(df_ids["IdCoup"])[0])
                )

    # Format : figure_list, key1 table contingence, key2 table contingence, id1, id2, data1, data2
    return (
        figure_list,
        pressions,
        labels_retourneur,
        [],
        flattened_rebonds_id_list,
        [],
        flattened_rebonds_list,
    )


def global_test(
    figure_list: list["Figure"],
    joueurs_list: list[str],
    study_selectionnee: str,
    effet_list: list = [],
    num_coups: list = [],
    random_state: int = 42,
) -> tuple[dict, list[list, list, list, list, list, list]]:
    """
    Récupère les retours de chaque joueur de joueurs_list sur tous leurs matchs de la base de données et effectue le traitement du chi2.
    Renvoie également les données utilisées pour les calculs.

    Args:
        figure_list (list): Liste des Figure sur laquelle tracer les points.
        joueurs_list (list): Liste des joueurs à traiter.
        study_selectionnee (str): Type d'étude à effectuer.
        effet_list (list, optionnel): Liste des effets à prendre en compte.
        num_coups (list, optionnel): Liste des numéros de coups.
        random_state (int, optionnel): Graine aléatoire pour le clustering.

    Returns:
        tuple: Un dictionnaire des résultats et une liste de listes de données utilisées.
    """

    resultats = {}  # Stocker les résultats pour chaque joueur.

    for indice, joueur in enumerate(joueurs_list):

        # Entre zone de retour et zone de service
        if study_selectionnee == "Retour_service":

            (
                figure_list,
                labels_serveur,
                labels_retourneur,
                flattened_precedent_id_list,
                flattened_rebonds_id_list,
                flattened_precedent_list,
                flattened_rebonds_list,
            ) = get_data_chi2_serves_returns(
                figure_list,
                num_coups,
                indice,
                joueur,
                effet_list,
                random_state,
            )

            if num_coups[0][indice] != 1:  # Pas le service
                # Calcul et stockage des tables de contingence et des p_values.
                table_contingence_result = table_contingence(
                    {"Service": labels_serveur, "Retour": labels_retourneur}
                )
                chi2_result = test_chi2(
                    {"Service": labels_serveur, "Retour": labels_retourneur}
                )

                resultats[joueur] = []
                resultats[joueur].append(
                    [table_contingence_result, chi2_result]
                )

            return resultats, [
                labels_serveur,
                labels_retourneur,
                flattened_precedent_id_list,
                flattened_rebonds_id_list,
                flattened_precedent_list,
                flattened_rebonds_list,
            ]

        # Entre retour et vainqueur du point en cours.
        if study_selectionnee == "Retour_vainqueur":
            (
                figure_list,
                flattened_vainqueurs,
                labels_retourneur,
                _,
                flattened_rebonds_id_list,
                _,
                flattened_rebonds_list,
            ) = get_data_chi2_winners_returns(
                figure_list,
                num_coups,
                indice,
                joueur,
                effet_list,
                random_state,
            )

            # Calcul et stockage des tables de contingence et des p_values.

            table_contingence_result = table_contingence(
                {
                    "vainqueurs": flattened_vainqueurs,
                    "labels_adv": labels_retourneur,
                }
            )
            chi2_result = test_chi2(
                {
                    "vainqueurs": flattened_vainqueurs,
                    "labels_adv": labels_retourneur,
                }
            )

            resultats[joueur] = []
            resultats[joueur].append([table_contingence_result, chi2_result])

            return resultats, [
                flattened_vainqueurs,
                labels_retourneur,
                [],
                flattened_rebonds_id_list,
                [],
                flattened_rebonds_list,
            ]

        # Entre retour et domination du point précédent.
        if study_selectionnee == "Retour_domination":
            (
                figure_list,
                flattened_domination,
                labels_retourneur,
                _,
                flattened_rebonds_id_list,
                _,
                flattened_rebonds_list,
            ) = get_data_chi2_domination_returns(
                figure_list, indice, joueur, random_state
            )

            # Calcul et stockage des tables de contingence et des p_values.

            table_contingence_result = table_contingence(
                {
                    "domination": flattened_domination,
                    "labels_adv": labels_retourneur,
                }
            )
            chi2_result = test_chi2(
                {
                    "domination": flattened_domination,
                    "labels_adv": labels_retourneur,
                }
            )

            resultats[joueur] = []
            resultats[joueur].append([table_contingence_result, chi2_result])

            return resultats, [
                flattened_domination,
                labels_retourneur,
                [],
                flattened_rebonds_id_list,
                [],
                flattened_rebonds_list,
            ]

        # Entre retour et pression.
        if study_selectionnee == "Retour_pression":
            (
                figure_list,
                flattened_pression,
                labels_retourneur,
                _,
                flattened_rebonds_id_list,
                _,
                flattened_rebonds_list,
            ) = get_data_chi2_pression_returns(
                figure_list, indice, joueur, random_state
            )

            # Calcul et stockage des tables de contingence et des p_values.

            table_contingence_result = table_contingence(
                {
                    "pression": flattened_pression,
                    "labels_adv": labels_retourneur,
                }
            )
            chi2_result = test_chi2(
                {
                    "pression": flattened_pression,
                    "labels_adv": labels_retourneur,
                }
            )

            resultats[joueur] = []
            resultats[joueur].append([table_contingence_result, chi2_result])

            return resultats, [
                flattened_pression,
                labels_retourneur,
                [],
                flattened_rebonds_id_list,
                [],
                flattened_rebonds_list,
            ]


def global_test_with_menu(joueurs: list[str], joueur_initial: str) -> None:
    """
    Affiche les données avec vidéos et ajoute un menu déroulant dont les options sont les joueurs.

    Args:
        joueurs (list): Liste des options du menu déroulant.
        joueur_initial (str): Choix par défaut dans le menu.

    Returns:
        None
    """
    # Créer un widget Dropdown (menu déroulant)
    dropdown = widgets.Dropdown(
        options=joueurs,
        value=joueur_initial,  # Valeur initiale
        description="Receveur :",
        layout=widgets.Layout(width="300px", height="50px"),
    )

    # Fonction pour gérer le changement de valeur dans le dropdown
    def on_dropdown_change(change):
        if change["type"] == "change" and change["name"] == "value":
            # Supprimer l'ancienne figure avant d'afficher la nouvelle
            clear_output(wait=True)
            # Réafficher le dropdown
            display(dropdown)
            # Créer une nouvelle figure et l'afficher avec la nouvelle valeur sélectionnée
            new_figure = Figure(False)
            # Récupérer la nouvelle valeur du menu
            joueur_selectionne = change["new"]
            global_test([new_figure], [joueur_selectionne])

    # Associer la fonction de rappel au Dropdown
    dropdown.observe(on_dropdown_change, names="value")

    # Afficher le dropdown au début
    display(dropdown)

    # Afficher la figure initiale avec le joueur par défaut
    initial_figure = Figure(False)
    global_test([initial_figure], [joueur_initial])

    return


def global_test_with_menu_dash(
    joueurs: list[str], joueur_initial: str
) -> None:
    """
    Gère l'analyse des retours avec une interface Dash, en utilisant `show_with_video()` pour afficher les résultats.

    Args:
        joueurs (list): Liste des options du menu déroulant.
        joueur_initial (str): Choix par défaut dans le menu.

    Returns:
        None
    """
    # Créer une nouvelle figure pour commencer
    initial_figure = Figure(False)
    effets = initial_figure.get_list_effets()

    # Appeler `show_with_video()` pour lancer l'application avec le dropdown pour sélectionner un joueur
    initial_figure.show_global_test_with_video(joueurs, joueur_initial, effets)


def sigmoid(x):
    """Applique la fonction sigmoïde pour garder la valeur entre 0 et 1."""
    return 1 / (1 + np.exp(-x))


def recup_services_reussis(nom_joueur: str, match: str):
    """
    Retourne une liste où un élément vaut True si le service est bon et que ce n'est pas un ace, et False sinon.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste de booléens indiquant si le service est réussi (hors ace).
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete_service = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {1}
            AND Gamename = '{match}'
        """
        df_service = pd.read_sql_query(requete_service, conn)
    finally:
        conn.close()
    service = [
        (df_service["coor_balle_x"][i], df_service["coor_balle_y"][i])
        for i in range(len(df_service["coor_balle_x"]))
    ]
    faute = [
        df_service["faute"][i] for i in range(len(df_service["coor_balle_x"]))
    ]
    # Liste qui vaut TRUE si le service est bon et FALSE sinon. Problème dans la base de données à gérer aussi.
    service_reussi = []
    for i in range(len(service)):
        if not math.isnan(service[i][0]) and faute[i] is None:
            service_reussi.append(True)
        elif math.isnan(service[i][0]) and faute[i] is None:
            service_reussi.append(True)
        else:
            service_reussi.append(False)
    return service_reussi


def recup_retours_reussis(nom_joueur: str, match: str):
    """
    Retourne une liste où le premier élément est une liste de booléens indiquant si le retour est bon, le deuxième une liste des joueurs qui retournent, et le troisième une liste des coordonnées.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: [liste de booléens, liste des joueurs, liste des coordonnées]
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete_retour = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {2}
            AND Gamename = '{match}'
        """
        df_retour = pd.read_sql_query(requete_retour, conn)
    finally:
        conn.close()
    retour = [
        (df_retour["coor_balle_x"][i], df_retour["coor_balle_y"][i])
        for i in range(len(df_retour["pos_balle_x_prece"]))
    ]
    coord = []  # Listes des coordonnées qui sera filtrée
    faute = [
        df_retour["faute"][i] for i in range(len(df_retour["coor_balle_x"]))
    ]
    qui_retour = [
        df_retour["joueur_frappe"][i]
        for i in range(len(df_retour["joueur_frappe"]))
    ]
    # Liste qui vaut TRUE si le retour est bon et FALSE sinon
    retour_reussi = []

    for i in range(len(retour)):
        coord.append(retour[i])
        if not math.isnan(retour[i][0]):
            retour_reussi.append(True)
        elif math.isnan(retour[i][0]) and faute[i] is None:
            retour_reussi.append(True)
        else:
            retour_reussi.append(False)
    return [retour_reussi, qui_retour, coord]


def data_domination(nom_joueur: str, match: str):
    """
    Retourne la liste des valeurs de domination en fonction du temps, ainsi que d'autres informations utiles pour l'analyse.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        tuple: (domination, qui_domine, domination_si_retour, coord, adjusted_labels)
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete = f"""
            SELECT * 
            FROM Liste_des_coups
            WHERE num_coup = {1}
            AND Gamename = '{match}'
        """
        df = pd.read_sql_query(requete, conn)
    finally:
        conn.close()

    retour_reussi = recup_retours_reussis(nom_joueur, match)[0]
    qui_retour = recup_retours_reussis(nom_joueur, match)[1]
    coord = recup_retours_reussis(nom_joueur, match)[2]
    coord = [
        coord[i]
        for i in range(len(coord))
        if qui_retour[i] == nom_joueur and retour_reussi[i] is True
    ]  # On récupère que les retours du joueur
    _, _, labels, _ = get_clusters_all_sets([], coord)
    adjusted_labels = [
        -1,
        -1,
    ]  # Premiers points où la domination n'est pas définie
    indice = 0
    for i, joueur in enumerate(qui_retour):
        if retour_reussi[i]:
            if joueur == nom_joueur:
                adjusted_labels.append(labels[indice])
                indice += 1
            else:
                adjusted_labels.append(-1)
        else:
            adjusted_labels.append(-1)

    service_reussi = recup_services_reussis(nom_joueur, match)

    domination = [0.5, 0.5, 0.5, 0.5]
    winner = [df["winner"][i] for i in range(len(df["coor_balle_x"]))]
    if not winner or not retour_reussi or not qui_retour:
        return [], [], []

    V = (
        []
    )  # Liste décrivant qui gagne le point à l'instant t : =1 si c'est nom_joueur, -1 sinon
    for i in range(len(winner)):
        if winner[i] == nom_joueur:
            V.append(1)
        else:
            V.append(-1)
    for i in range(4, len(df["coor_balle_x"])):
        beta = []
        p = 2
        for j in range(i):
            beta.append(5 * j**p / sum([k**p for k in range(i)]))
        domination_value = sum([beta[i - j] * V[i - j] for j in range(1, i)])
        domination.append(sigmoid(domination_value))
    # Récupère la liste du score de domination que pour les points où le retour est bon
    # On trie d'abord sur les services puis sur les retours
    domination_si_service = [
        domination[i]
        for i in range(len(domination))
        if service_reussi[i] is True
    ]

    domination_si_retour = [
        domination_si_service[i]
        for i in range(len(domination_si_service))
        if retour_reussi[i] is True and qui_retour[i] == nom_joueur
    ]
    # Avoir la liste du nom de celui qui domine
    qui_domine = []
    for score in domination_si_retour:
        if score >= 0.6:
            qui_domine.append(nom_joueur)
        elif 0.4 < score < 0.6:
            qui_domine.append("Personne")
        else:
            qui_domine.append("Adveraire")

    return domination, qui_domine, domination_si_retour, coord, adjusted_labels


def plot_domination_score_evolution(player_name: str, match: str):
    """
    Affiche l'évolution du score, de la domination et des clusters au cours d'un match de tennis de table.

    Args:
        player_name (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        None
    """
    # Retrieve the data
    domination, _, _, _, labels = data_domination(player_name, match)
    score_player, score_opponent = evolution_score(player_name, match)
    t = list(range(len(domination)))

    # Initialize the lists for played points
    points_player = []
    points_opponent = []

    for set_name in score_player:
        points_player.extend(score_player[set_name])
        points_opponent.extend(score_opponent[set_name])
    set_endings = [len(score_player[set_name]) for set_name in score_player]
    cumulative_points = 0

    # Configuration for LaTeX-style fonts
    rc("font", **{"family": "serif", "serif": ["Latin Modern Math"]})
    rcParams["mathtext.fontset"] = "cm"

    # Create the figure with a slightly increased vertical spacing
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 12), gridspec_kw={"hspace": 0.1}
    )

    # First plot: Score evolution
    ax1.plot(
        points_player,
        "b-",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=r"Alexis Lebrun",
    )
    ax1.plot(
        points_opponent,
        "green",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=r"Opponent",
    )
    for i, end in enumerate(set_endings):
        cumulative_points += end
        if i == 0:
            ax1.axvline(
                x=cumulative_points,
                linestyle="--",
                color="gray",
                alpha=0.5,
                label="End of set",
            )
        else:
            ax1.axvline(
                x=cumulative_points, linestyle="--", color="gray", alpha=0.5
            )
        ax1.text(
            cumulative_points,
            min(min(points_player), min(points_opponent)) - 1,
            f"End of {list(score_player.keys())[i]}",
            rotation=0,
            verticalalignment="top",
            horizontalalignment="center",
            fontsize=10,
        )
    ax1.set_ylabel(r"Scores", fontsize=14)
    ax1.set_title(
        r"Score, Domination and Cluster Label Evolution - Table Tennis Match",
        fontsize=16,
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Second plot: Domination
    ax2.plot(
        t,
        domination,
        marker=".",
        markersize=5,
        linewidth=1.5,
        label="Domination",
        color="orange",
    )
    ax2.axhline(
        y=0.5, color="red", linestyle="--", label="Equilibrium Threshold"
    )
    cumulative_points = 0
    for i, end in enumerate(set_endings):
        cumulative_points += end
        if i == 0:
            ax2.axvline(
                x=cumulative_points,
                linestyle="--",
                color="gray",
                alpha=0.5,
                label="End of set",
            )
        else:
            ax2.axvline(
                x=cumulative_points, linestyle="--", color="gray", alpha=0.5
            )
    ax2.set_ylabel(r"Domination Score ($D_t$)", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 1. Convertir en float et identifier les -1
    labels_arr = np.array(labels, dtype=float)
    mask_missing = labels_arr == -1

    # 2. Préparer les données pour les marqueurs : NaN aux endroits manquants
    labels_markers = labels_arr.copy()
    labels_markers[mask_missing] = np.nan

    # 3. Interpoler linéairement pour combler les -1 et avoir une courbe continue
    valid_idx = np.where(~mask_missing)[0]
    labels_interp = np.interp(
        x=np.arange(len(labels_arr)), xp=valid_idx, fp=labels_arr[valid_idx]
    )

    # 4. Tracer d’abord la ligne continue (sans marqueurs)
    ax3.plot(
        t,
        labels_interp,
        linestyle="-",
        linewidth=1.5,
        label="Clusters",
        color="black",
    )

    # 5. Ajouter ensuite les points (marqueurs) uniquement là où y != -1
    ax3.plot(
        t,
        labels_markers,
        linestyle="None",
        marker=".",
        markersize=5,
        color="black",
    )

    # 6. Lignes verticales « End of set »
    cumulative_points = 0
    for i, end in enumerate(set_endings):
        cumulative_points += end
        if i == 0:
            ax3.axvline(
                x=cumulative_points,
                linestyle="--",
                color="gray",
                alpha=0.5,
                label="End of set",
            )
        else:
            ax3.axvline(
                x=cumulative_points, linestyle="--", color="gray", alpha=0.5
            )

    # 7. Axe Y à valeurs entières
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 8. Labels, légende et grille
    ax3.set_ylabel(r"Cluster Label", fontsize=14)
    ax3.set_xlabel(r"Points Played", fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    # plt.savefig("match_lebun_zhendong.eps", bbox_inches="tight")
    plt.show()


def plot_pression_score_evolution(player_name: str, match: str):
    """
    Affiche l'évolution du score, de la pression et des clusters au cours d'un match de tennis de table.

    Args:
        player_name (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        None
    """
    # Retrieve the data
    pression = pression_totale(player_name, match)
    _, _, _, _, labels = data_domination(player_name, match)
    score_player, score_opponent = evolution_score(player_name, match)
    t = list(range(len(pression)))

    # Initialize the lists for played points
    points_player = []
    points_opponent = []

    for set_name in score_player:
        points_player.extend(score_player[set_name])
        points_opponent.extend(score_opponent[set_name])
    set_endings = [len(score_player[set_name]) for set_name in score_player]
    cumulative_points = 0

    # Configuration for LaTeX-style fonts
    rc("font", **{"family": "serif", "serif": ["Latin Modern Math"]})
    rcParams["mathtext.fontset"] = "cm"

    # Create the figure with a slightly increased vertical spacing
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 12), gridspec_kw={"hspace": 0.1}
    )

    # First plot: Score evolution
    ax1.plot(
        points_player,
        "b-",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=r"Alexis Lebrun",
    )
    ax1.plot(
        points_opponent,
        "green",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label=r"Opponent",
    )
    for i, end in enumerate(set_endings):
        cumulative_points += end
        if i == 0:
            ax1.axvline(
                x=cumulative_points,
                linestyle="--",
                color="gray",
                alpha=0.5,
                label="End of set",
            )
        else:
            ax1.axvline(
                x=cumulative_points, linestyle="--", color="gray", alpha=0.5
            )
        ax1.text(
            cumulative_points,
            min(min(points_player), min(points_opponent)) - 1,
            f"End of {list(score_player.keys())[i]}",
            rotation=0,
            verticalalignment="top",
            horizontalalignment="center",
            fontsize=10,
        )
    ax1.set_ylabel(r"Scores", fontsize=14)
    ax1.set_title(
        r"Score, Pression and Cluster Label Evolution - Table Tennis Match",
        fontsize=16,
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Second plot: Domination
    ax2.plot(
        t,
        pression,
        marker=".",
        markersize=5,
        linewidth=1.5,
        label="Pression",
        color="orange",
    )
    cumulative_points = 0
    for i, end in enumerate(set_endings):
        cumulative_points += end
        if i == 0:
            ax2.axvline(
                x=cumulative_points,
                linestyle="--",
                color="gray",
                alpha=0.5,
                label="End of set",
            )
        else:
            ax2.axvline(
                x=cumulative_points, linestyle="--", color="gray", alpha=0.5
            )
    ax2.set_ylabel(r"Pression Score ($P_t$)", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Third plot: clusters
    ax3.plot(
        t,
        labels,
        marker=".",
        markersize=5,
        linewidth=1.5,
        label="Clusters",
        color="black",
    )

    cumulative_points = 0
    for i, end in enumerate(set_endings):
        cumulative_points += end
        if i == 0:
            ax3.axvline(
                x=cumulative_points,
                linestyle="--",
                color="gray",
                alpha=0.5,
                label="End of set",
            )
        else:
            ax3.axvline(
                x=cumulative_points, linestyle="--", color="gray", alpha=0.5
            )

    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax3.set_ylabel(r"Cluster Label", fontsize=14)
    ax3.set_xlabel(r"Points Played", fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    # plt.savefig("match_lebun_zhendong_pression.eps", bbox_inches="tight")
    plt.show()


# Représente l'évolution de la domination pendant le match
def plot_domination_evolution(nom_joueur: str, match: str):
    """
    Représente graphiquement l'évolution de la domination pendant le match.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        None
    """
    domination = data_domination(nom_joueur, match)[0]
    # Définir les valeurs de t de 0 à len(domination)
    t = list(range(len(domination)))

    # Tracer la liste de domination
    plt.figure(figsize=(10, 6))
    plt.plot(t, domination, marker="o", linestyle="-", color="b")
    # Ajouter une ligne constante pour D=0.5
    plt.axhline(y=0.5, color="r", linestyle="--", label="D = 0.5 (Constante)")
    # Ajouter des labels et un titre
    plt.title("Évolution de la Domination au Cours du Match")
    plt.xlabel("Temps (t)")
    plt.ylabel("Score de Domination (D_t)")
    plt.xticks(t, fontsize=5)  # Pour avoir des ticks pour chaque valeur de t
    plt.ylim(0, 1)  # Limiter l'axe y entre 0 et 1
    plt.grid()  # Ajouter une grille pour la lisibilité

    # Afficher le graphique
    plt.show()


# Récupère l'évolution du score en fonction des points et des sets
def evolution_score(nom_joueur: str, match: str):
    """
    Récupère l'évolution du score en fonction des points et des sets.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        tuple: (dico_joueur, dico_adv) Dictionnaires des scores par set pour le joueur et l'adversaire.
    """
    conn = sqlite3.connect(chemin_bdd)
    try:
        requete = f"""
            SELECT Liste_des_points.IdPoint AS IdPoint, Liste_des_points.numero_set AS numero_set, Liste_des_points.point_pour AS point_pour,
                    Liste_des_points.score_jA AS score_jA, Liste_des_points.score_jB AS score_jB
            FROM Liste_des_points
            JOIN Liste_des_coups
            ON Liste_des_points.IdMatch = Liste_des_coups.IdMatch
            WHERE  Gamename = '{match}'
            GROUP BY Liste_des_points.IdPoint
        """
        df = pd.read_sql_query(requete, conn)
    finally:
        conn.close()
    liste_numero_set = [df["numero_set"][i] for i in range(len(df["IdPoint"]))]
    # Détecter si nom_joueur est le joueurA ou le joueurB
    score_joueur = None
    score_adv = None
    point0_pour = df["point_pour"][0]
    if point0_pour == nom_joueur:
        if df["score_jA"].tolist()[1] == 1:
            score_joueur = "score_jA"
            score_adv = "score_jB"
        else:
            score_joueur = "score_jB"
            score_adv = "score_jA"
    else:
        if df["score_jA"].tolist()[1] == 1:
            score_joueur = "score_jB"
            score_adv = "score_jA"
        else:
            score_joueur = "score_jA"
            score_adv = "score_jB"

    liste_score_joueur = [
        df[score_joueur][i] for i in range(len(df["IdPoint"]))
    ]
    liste_score_adv = [df[score_adv][i] for i in range(len(df["IdPoint"]))]

    dico_joueur = (
        {}
    )  # Dictionnaire qui prend en clé le numéro du set, et en valeur l'avancement du score du joueur dans ce set et le vainqueur du set
    dico_adv = {}
    for i in range(1, max(liste_numero_set) + 1):
        dico_joueur[f"Set {i}"] = []
        dico_adv[f"Set {i}"] = []
    for i in range(len(liste_numero_set)):
        if liste_numero_set[i] == 1:
            dico_joueur[f"Set {1}"].append(liste_score_joueur[i])
            dico_adv[f"Set {1}"].append(liste_score_adv[i])
        if liste_numero_set[i] == 2:
            dico_joueur[f"Set {2}"].append(liste_score_joueur[i])
            dico_adv[f"Set {2}"].append(liste_score_adv[i])
        if liste_numero_set[i] == 3:
            dico_joueur[f"Set {3}"].append(liste_score_joueur[i])
            dico_adv[f"Set {3}"].append(liste_score_adv[i])
        if liste_numero_set[i] == 4:
            dico_joueur[f"Set {4}"].append(liste_score_joueur[i])
            dico_adv[f"Set {4}"].append(liste_score_adv[i])
        if liste_numero_set[i] == 5:
            dico_joueur[f"Set {5}"].append(liste_score_joueur[i])
            dico_adv[f"Set {5}"].append(liste_score_adv[i])
    return dico_joueur, dico_adv


# Représenter graphiquement l'évolution des points en fonction du temps


def affiche_evolution_score(nom_joueur: str, match: str):
    """
    Représente graphiquement l'évolution des points en fonction du temps.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        None
    """
    score_joueur, score_adv = evolution_score(nom_joueur, match)

    # Configuration des graphiques
    plt.figure(figsize=(12, 6))

    # Initialisation des listes pour les points joués
    points_joueur = []
    points_adv = []

    # Traçage des scores des deux joueurs
    for set_name in score_joueur:
        points_joueur.extend(score_joueur[set_name])
        points_adv.extend(score_adv[set_name])

    plt.plot(points_joueur, label=nom_joueur, color="blue", marker="o")
    plt.plot(points_adv, label="Adversaire", color="orange", marker="o")

    # Marquer la fin de chaque set
    set_endings = [len(score_joueur[set_name]) for set_name in score_joueur]
    cumulative_points = 0
    for i, end in enumerate(set_endings):
        cumulative_points += end
        plt.axvline(
            x=cumulative_points, linestyle="--", color="gray", alpha=0.5
        )
        plt.text(
            cumulative_points,
            max(max(points_joueur), max(points_adv)) + 1,
            f"Fin {list(score_joueur.keys())[i]}",
            rotation=0,
            verticalalignment="center",
            horizontalalignment="right",
        )

    # Axes et titres
    plt.title("Évolution des scores - Match de Tennis de Table")
    plt.xlabel("Points Joués")
    plt.ylabel("Scores")
    plt.xticks(
        range(len(points_joueur)), fontsize=5
    )  # Ajuster les ticks pour tous les points
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(max(points_joueur), max(points_adv)) + 2)

    # Affichage du graphique
    plt.show()


# Mesure l'évolution de la pression pendant le match


def pression_score(nom_joueur: str, match: str):
    """
    Retourne l'indicateur de pression lié à l'écart de points dans les sets.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste des indicateurs de pression pour chaque point.
    """
    dico_score_joueur, dico_score_adv = evolution_score(nom_joueur, match)
    liste_score_joueur = []
    liste_score_adv = []
    for liste_score in dico_score_joueur.values():
        for score in liste_score:
            liste_score_joueur.append(score)
    for liste_score in dico_score_adv.values():
        for score in liste_score:
            liste_score_adv.append(score)
    liste_pression = []
    for i in range(len(liste_score_joueur)):
        liste_pression.append(
            1 / (1 + abs(liste_score_joueur[i] - liste_score_adv[i]))
        )
    return liste_pression


def pression_moments_clés(nom_joueur: str, match: str):
    """
    Retourne l'indicateur de pression lié aux moments clés (balle de match ou de set à des moments serrés).

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste des indicateurs de pression pour chaque point.
    """
    dico_score_joueur, dico_score_adv = evolution_score(nom_joueur, match)
    liste_score_joueur = []
    liste_score_adv = []
    for liste_score in dico_score_joueur.values():
        for score in liste_score:
            liste_score_joueur.append(score)
    for liste_score in dico_score_adv.values():
        for score in liste_score:
            liste_score_adv.append(score)
    liste_pression = []
    for i in range(len(liste_score_joueur)):
        if 2 >= abs(liste_score_joueur[i] - liste_score_adv[i]) >= 1 and max(
            liste_score_joueur[i], liste_score_adv[i]
        ) in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            liste_pression.append(1)
        else:
            liste_pression.append(0)
    return liste_pression


def pression_fin_set(nom_joueur: str, match: str):
    """
    Retourne l'indicateur de pression lié à la proximité de la fin du set.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste des indicateurs de pression pour chaque point.
    """
    dico_score_joueur, dico_score_adv = evolution_score(nom_joueur, match)
    liste_score_joueur = []
    liste_score_adv = []
    for liste_score in dico_score_joueur.values():
        for score in liste_score:
            liste_score_joueur.append(score)
    for liste_score in dico_score_adv.values():
        for score in liste_score:
            liste_score_adv.append(score)
    liste_pression = []
    for i in range(len(liste_score_joueur)):
        liste_pression.append(
            1
            / (
                1
                + abs(
                    10
                    - min(10, max(liste_score_joueur[i], liste_score_adv[i]))
                )
            )
        )
    return liste_pression


def pression_score_set(nom_joueur: str, match: str):
    """
    Retourne l'indicateur de pression lié au score en termes de set.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste des indicateurs de pression pour chaque point.
    """
    dico_score_joueur, dico_score_adv = evolution_score(nom_joueur, match)
    n_sets = len(dico_score_joueur.keys())
    score_set_joueur = [0]
    score_set_adv = [0]
    for i in range(1, n_sets):
        if (
            dico_score_joueur[list(dico_score_joueur.keys())[i - 1]][-1]
            > dico_score_adv[list(dico_score_adv.keys())[i - 1]][-1]
        ):
            score_set_joueur.append(score_set_joueur[i - 1] + 1)
            score_set_adv.append(score_set_adv[i - 1])
        else:
            score_set_adv.append(score_set_adv[i - 1] + 1)
            score_set_joueur.append(score_set_joueur[i - 1])
    liste_pression = []
    for i in range(n_sets):
        for j in range(
            len(dico_score_joueur[list(dico_score_joueur.keys())[i]])
        ):
            liste_pression.append(
                1 / (1 + abs(score_set_joueur[i] - score_set_adv[i]))
            )
    return liste_pression


def pression_set_decisif(nom_joueur: str, match: str):
    """
    Retourne l'indicateur de pression lié à la présence d'un set décisif.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste des indicateurs de pression pour chaque point.
    """
    dico_score_joueur, dico_score_adv = evolution_score(nom_joueur, match)
    n_sets = len(dico_score_joueur.keys())
    score_set_joueur = [0]
    score_set_adv = [0]
    for i in range(1, n_sets):
        if (
            dico_score_joueur[list(dico_score_joueur.keys())[i - 1]][-1]
            > dico_score_adv[list(dico_score_adv.keys())[i - 1]][-1]
        ):
            score_set_joueur.append(score_set_joueur[i - 1] + 1)
            score_set_adv.append(score_set_adv[i - 1])
        else:
            score_set_adv.append(score_set_adv[i - 1] + 1)
            score_set_joueur.append(score_set_joueur[i - 1])
    liste_pression = []
    for i in range(n_sets):
        for j in range(
            len(dico_score_joueur[list(dico_score_joueur.keys())[i]])
        ):
            if score_set_adv[i] == 2 and score_set_joueur[i] == 2:
                liste_pression.append(1)
            else:
                liste_pression.append(0)
    return liste_pression


def pression_totale(nom_joueur: str, match: str):
    """
    Retourne l'évolution de la pression totale pendant le match.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste de la pression totale pour chaque point.
    """
    p1 = 2.25
    p2 = 2.25
    p3 = 2.25
    p4 = 0.75
    p5 = 2.5

    press_score = pression_score(nom_joueur, match)
    press_clés = pression_moments_clés(nom_joueur, match)
    press_fin_set = pression_fin_set(nom_joueur, match)
    press_score_set = pression_score_set(nom_joueur, match)
    press_set_decisif = pression_set_decisif(nom_joueur, match)
    liste_pression = []
    for i in range(len(press_score)):
        pression = (
            p1 * press_score[i]
            + p2 * press_clés[i]
            + p3 * press_fin_set[i]
            + p4 * press_score_set[i]
            + p5 * press_set_decisif[i]
        )
        liste_pression.append(pression)
    return liste_pression


def seuils_pression(nom_joueur: str, match: str):
    """
    Retourne les seuils a, b et c tels que 33%, 66% et 85% des données de pression soient inférieurs à ces valeurs.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste des seuils [a, b, c].
    """
    liste_pression = pression_totale(nom_joueur, match)
    a = np.percentile(liste_pression, 33)  # 33e percentile
    b = np.percentile(liste_pression, 66)
    c = np.percentile(liste_pression, 85)
    return [a, b, c]


def liste_pression_seuil(nom_joueur: str, match: str):
    """
    Associe chaque indicateur de pression à une zone de pression en fonction des seuils, seulement pour les points où nom_joueur est au retour.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        list: Liste des zones de pression ("Faible", "Moyenne", "Assez forte", "Forte").
    """
    liste_pression = pression_totale(nom_joueur, match)
    liste_seuils = seuils_pression(nom_joueur, match)
    a, b, c = liste_seuils[0], liste_seuils[1], liste_seuils[2]
    new_liste = []
    for i in range(len(liste_pression)):
        if liste_pression[i] < a:
            new_liste.append("Faible")
        if a <= liste_pression[i] < b:
            new_liste.append("Moyenne")
        if b <= liste_pression[i] < c:
            new_liste.append("Assez forte")
        if c <= liste_pression[i]:
            new_liste.append("Forte")
    return new_liste


def recup_pression_si_retour(nom_joueur: str, match: str):
    """
    Retourne la pression seulement pour les points où le service est bon, le retour est bon et effectué par nom_joueur.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        tuple: (pression_si_retour, coord) Liste des pressions et coordonnées associées.
    """
    pression = liste_pression_seuil(nom_joueur, match)
    service_reussi = recup_services_reussis(nom_joueur, match)
    retour_reussi = recup_retours_reussis(nom_joueur, match)[0]
    qui_retour = recup_retours_reussis(nom_joueur, match)[1]

    coord = recup_retours_reussis(nom_joueur, match)[2]
    coord = [
        coord[i]
        for i in range(len(coord))
        if qui_retour[i] == nom_joueur and retour_reussi[i] is True
    ]  # On récupère que les retours du joueur

    # On trie d'abord sur les services puis sur les retours
    pression_si_service = [
        pression[i] for i in range(len(pression)) if service_reussi[i] is True
    ]
    pression_si_retour = [
        pression_si_service[i]
        for i in range(len(pression_si_service))
        if retour_reussi[i] is True and qui_retour[i] == nom_joueur
    ]
    return pression_si_retour, coord


# Représente graphiquement l'évolution de la pression pendant le match
def plot_pression_evolution(nom_joueur: str, match: str):
    """
    Représente graphiquement l'évolution de la pression pendant le match.

    Args:
        nom_joueur (str): Nom du joueur.
        match (str): Nom du match.

    Returns:
        None
    """
    pression = pression_totale(nom_joueur, match)
    # Définir les valeurs de t de 0 à len(domination)
    t = list(range(len(pression)))

    # Tracer la liste de domination
    plt.figure(figsize=(12, 6))
    plt.plot(t, pression, marker="o", linestyle="-", color="b")
    # Ajouter des labels et un titre
    plt.title("Évolution de la Pression au Cours du Match")
    plt.xlabel("Temps (t)")
    plt.ylabel("Indicateur de Pression (P_t)")
    plt.xticks(t, fontsize=5)  # Pour avoir des ticks pour chaque valeur de t
    plt.grid()  # Ajouter une grille pour la lisibilité

    # Afficher le graphique
    plt.show()
