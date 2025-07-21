import math
import os
import socket
import sqlite3
import sys

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dash_table, dcc, html
from flask import send_from_directory

from modules.data_processing import get_clusters_all_sets, place_on_side
from modules.figure_maker import (
    draw_clusters,
    draw_joueurs,
    draw_rebonds,
    draw_table,
)

chemin_dossier_parent = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(chemin_dossier_parent)
chemin_bdd = os.path.join(chemin_dossier_parent, "BDD_avec_cluster.db")
chemin_dossier_images = os.path.join(os.getcwd(), "images")


def find_free_port():
    """
    Donne un numéro de port libre pour l'utilisation des applications dash.

    Returns:
        int: Numéro de port libre.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


class Figure:
    """
    L'objet `Figure` permet d'afficher plusieurs listes de rebonds, de positions de joueur et de clusters.
    On a aussi la possibilité de tracer les trajectoires entre les rebonds d'un cluster et les rebonds précédents correspondants.
    On peut lancer une application Dash permettant la visualisation des figures, des vidéos des points, des tables de contingence, des tests du chi2, etc.
    """

    def __init__(self, show_grid: bool = False):
        """
        Construit la classe Figure.

        Args:
            show_grid (bool): Indique si l'on souhaite tracer un quadrillage sur la table.
        """
        self.fig = draw_table(show_grid)
        self.positions_rebonds_list = []
        self.positions_joueur_list = []
        self.clusters_list = []  # Vérifier l'utilité

    def get_fig(self) -> "Figure":
        """
        Retourne la figure Plotly associée à l'objet Figure.

        Returns:
            Figure: Objet Plotly de la figure.
        """
        return self.fig

    def get_positions_rebonds_list(self) -> list[tuple[float, float]]:
        """
        Retourne la liste des positions de rebonds ajoutées à la figure.

        Returns:
            list: Liste des positions de rebonds.
        """
        return self.positions_rebonds_list

    def get_positions_joueur_list(self) -> list[tuple[float, float]]:
        """
        Retourne la liste des positions de joueur ajoutées à la figure.

        Returns:
            list: Liste des positions de joueur.
        """
        return self.positions_joueur_list

    def get_clusters_list(self) -> list:
        """
        Retourne la liste des clusters ajoutés à la figure.

        Returns:
            list: Liste des clusters.
        """
        return self.clusters_list

    def add_positions_rebonds(
        self,
        positions_rebonds_list: list[tuple[float, float]],
        num_coup_list: list[int],
        colors: list = None,
    ) -> None:
        """
        Ajoute les rebonds dans toutes les listes de rebonds de positions_rebonds_list.
        La fonction calcule automatiquement le côté du joueur en fonction du numéro de coup.

        Args:
            positions_rebonds_list (list): Liste de liste de coordonnées de rebonds.
            num_coup_list (list): Liste des numéros de coup correspondant à chaque liste de positions_rebonds_list.
            colors (list, optionnel): Liste des couleurs à appliquer sur les points.

        Returns:
            None
        """
        # Initialiser colors à une liste vide si None
        if colors is None:
            colors = []

        # Calculer la liste des joueurs une seule fois
        joueur_list = [0 if num % 2 == 1 else 1 for num in num_coup_list]

        for i in range(len(positions_rebonds_list)):
            self.positions_rebonds_list.append(positions_rebonds_list[i])

            if colors and i < len(colors):
                # Utiliser les couleurs fournies
                draw_rebonds(
                    self.fig,
                    positions_rebonds_list[i],
                    joueur_list[i],
                    colors=colors[i],
                )
            else:
                # # Calculer les couleurs en fonction des gagnants
                # winners = get_colors_win_lose(positions_rebonds_list[i])

                # print(winners)

                # # Création de la liste des couleurs pour cette position
                # point_colors = [
                #     (
                #         "green"
                #         if w and w[1] == "returner"
                #         else "red" if w and w[1] == "server" else "white"
                #     )
                #     for w in winners
                # ]

                draw_rebonds(
                    self.fig,
                    positions_rebonds_list[i],
                    joueur_list[i],
                    # colors=point_colors,
                )
        return

    def add_positions_joueur(
        self,
        positions_joueur_list: list[tuple[float, float]],
        num_coup_list: list[int],
    ) -> None:
        """
        Ajoute les positions de joueur dans toutes les listes de positions_joueur_list.
        La fonction prend en compte le côté sur lequel doivent être placés les points grâce à num_coup_list indiquant à quels numéros de coups correspondent chaque liste de positions de joueur.

        Args:
            positions_joueur_list (list): Liste de liste de coordonnées de positions de joueur.
            num_coup_list (list): Liste des numéros de coups correspondant à chaque liste de positions_joueur_list.

        Returns:
            None
        """
        # On déduit le numéro du joueur selon la parité du coup (0 pour serveur, 1 pour retourneur).
        joueur_list = [
            0 if num_coup_list[i] % 2 == 1 else 1
            for i in range(len(num_coup_list))
        ]
        for i in range(len(positions_joueur_list)):
            self.positions_joueur_list.append(positions_joueur_list[i])
            draw_joueurs(self.fig, positions_joueur_list[i], joueur_list[i])
        return

    def add_clusters(
        self,
        positions_rebonds_list: list[tuple[float, float]],
        k_list: list[int],
        num_coup_list: list[int],
        random_state: int = 42,
    ) -> list[int]:
        """
        Trace l'ensemble des clusters pour chaque liste de rebonds de positions_rebonds_list.
        Le nombre de clusters est donné par k_list, et le côté est indiqué par num_coup_list.

        Args:
            positions_rebonds_list (list): Liste de liste de coordonnées de rebonds.
            k_list (list): Liste du nombre de clusters à utiliser dans l'algorithme des k-moyennes pour chaque liste.
            num_coup_list (list): Liste des numéros de coups correspondant à chaque liste de positions_rebonds_list.
            random_state (int, optionnel): Graine aléatoire pour le clustering.

        Returns:
            list: La liste des labels calculés.
        """
        # On déduite le numéro du joueur selon la parité du coup (0 pour serveur, 1 pour retourneur).
        joueur_list = [
            0 if num_coup_list[i] % 2 == 1 else 1
            for i in range(len(num_coup_list))
        ]
        labels_list = []
        # Place les clusters du bon côté
        for i in range(len(positions_rebonds_list)):
            if joueur_list[i] == 0:  # Serveur
                _, positions_rebonds_list[i], _, _ = place_on_side(
                    [], positions_rebonds_list[i], [], []
                )
                labels, enveloppes, _, _ = get_clusters_all_sets(
                    positions_rebonds_list[i], [], k_list[i], 0, random_state
                )
                labels_list.append(labels)
            if joueur_list[i] == 1:  # Retourneur
                _, _, _, positions_rebonds_list[i] = place_on_side(
                    [], [], [], positions_rebonds_list[i]
                )
                [], [], labels, enveloppes = get_clusters_all_sets(
                    [], positions_rebonds_list[i], 0, k_list[i], random_state
                )
                labels_list.append(labels)

            self.clusters_list.append(positions_rebonds_list[i])
            draw_clusters(
                self.fig,
                enveloppes,
                labels,
                positions_rebonds_list[i],
                k_list[i],
            )

        return labels_list

    def trajectories(self, list1: list, list2: list) -> None:
        """
        Fonction auxiliaire pour relier les éléments de la liste list1 aux éléments de la liste list2.

        Args:
            list1 (list): Première liste d'éléments.
            list2 (list): Seconde liste d'éléments.

        Returns:
            None
        """
        # On oppose l'abscisse pour rendre correcte la symétrie.
        list1 = [(-list1[i][0], list1[i][1]) for i in range(len(list1))]
        list2 = [(-list2[i][0], list2[i][1]) for i in range(len(list2))]
        for i in range(len(list1)):
            self.fig.add_trace(
                go.Scatter(
                    x=[list1[i][0], list2[i][0]],
                    y=[list1[i][1], list2[i][1]],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name="Trajectoires",
                    showlegend=(i == 0),
                )
            )

        return None

    def trajectories_on_cluster(
        self,
        positions_rebonds_adv: list[tuple[float, float]],
        positions_rebonds_prece: list[tuple[float, float]],
        labels: list[int],
        cluster_to_link_to: int,
    ) -> None:
        """
        Relie l'ensemble des points appartenant au cluster cluster_to_link_to au rebond précédent correspondant.

        Args:
            positions_rebonds_adv (list): Liste des rebonds de l'adversaire formant l'ensemble des clusters.
            positions_rebonds_prece (list): Liste des rebonds précédents les rebonds dans positions_rebonds_adv.
            labels (list): Liste des labels des points de positions_rebonds_adv.
            cluster_to_link_to (int): Numéro du cluster vers lequel afficher les trajectoires.

        Returns:
            None
        """
        # On oppose l'abscisse pour rendre correcte la symétrie.
        positions_rebonds_adv = [
            (-positions_rebonds_adv[i][0], positions_rebonds_adv[i][1])
            for i in range(len(positions_rebonds_adv))
        ]
        positions_rebonds_prece = [
            (-positions_rebonds_prece[i][0], positions_rebonds_prece[i][1])
            for i in range(len(positions_rebonds_prece))
        ]

        assert cluster_to_link_to in labels  # Vérifie que le cluster existe.

        # Préparation des listes de rebonds.
        _, _, _, positions_rebonds_adv = place_on_side(
            [], [], [], positions_rebonds_adv
        )

        positions_rebonds_adv_x = [
            positions_rebonds_adv[i][0]
            for i in range(len(positions_rebonds_adv))
        ]
        positions_rebonds_adv_y = [
            positions_rebonds_adv[i][1]
            for i in range(len(positions_rebonds_adv))
        ]

        _, positions_rebonds_prece, _, _ = place_on_side(
            [], positions_rebonds_prece, [], []
        )
        positions_rebonds_prece_x = [
            positions_rebonds_prece[i][0]
            for i in range(len(positions_rebonds_prece))
        ]
        positions_rebonds_prece_y = [
            positions_rebonds_prece[i][1]
            for i in range(len(positions_rebonds_prece))
        ]

        # On filtre les trajectoires par cluster de l'adversaire. On récupère pour cela les indices dont le labels correspond à cluster_to_link_to.
        indices = [
            i
            for i in range(len(positions_rebonds_adv_x))
            if labels[i] == cluster_to_link_to
        ]

        positions_rebonds_prece_x = [
            positions_rebonds_prece[i][0] for i in indices
        ]
        positions_rebonds_prece_y = [
            positions_rebonds_prece[i][1] for i in indices
        ]

        positions_rebonds_cluster_adv_x = [
            positions_rebonds_adv_x[i] for i in indices
        ]
        positions_rebonds_cluster_adv_y = [
            positions_rebonds_adv_y[i] for i in indices
        ]

        # Affichage
        for i in range(len(indices)):
            self.fig.add_trace(
                go.Scatter(
                    x=[
                        positions_rebonds_cluster_adv_x[i],
                        positions_rebonds_prece_x[i],
                    ],
                    y=[
                        positions_rebonds_cluster_adv_y[i],
                        positions_rebonds_prece_y[i],
                    ],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name="Trajectoires",
                    showlegend=(i == 0),
                )
            )

        return None

    def show(self) -> None:
        """
        Affiche la figure de manière statique.

        Returns:
            None
        """
        self.fig.show()
        return None

    def show_data_with_video(self) -> None:
        """
        Affiche la figure encapsulée dans une application Dash permettant d'afficher la vidéo associée à un point de la figure.
        Un lien pour l'ouvrir dans le navigateur est généré.

        Returns:
            None
        """

        # Création de l'application
        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Création de la mise en page
        app.layout = html.Div(
            children=[
                # Affichage de la table de la vidéo
                html.Div(
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "center",
                        "align-items": "flex-start",
                    },
                    children=[
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="pingpong-table",
                                    figure=self.get_fig(),  # Utilise la figure déjà créée
                                    style={"height": "500px", "flex": "1"},
                                )
                            ],
                            style={"flex": "3", "margin-right": "10px"},
                        ),  # Réduit la marge à droite
                        html.Div(
                            id="video-container",
                            children=[
                                html.Video(
                                    id="video-player",
                                    controls=True,
                                    autoPlay=True,
                                    src="",
                                    style={
                                        "width": "600px",
                                        "max-width": "100%",
                                        "display": "none",
                                    },
                                )
                            ],
                            style={
                                "flex": "2",
                                "padding-left": "10px",  # Réduit l'écart à gauche
                                "height": "500px",
                                "display": "flex",
                                "align-items": "center",
                            },
                        ),
                    ],
                )
            ]
        )

        # Callback pour mettre à jour la vidéo en fonction du point cliqué
        @app.callback(
            [Output("video-player", "src"), Output("video-player", "style")],
            [Input("pingpong-table", "clickData")],
        )
        def display_video_on_click(clickData):
            if clickData is None:
                return "", {"display": "none"}

            try:
                # Récupérer les coordonnées du point cliqué
                x_clicked = clickData["points"][0]["x"]
                y_clicked = clickData["points"][0]["y"]
                # print(f"Coordonnées cliquées : x={x_clicked}, y={y_clicked}")

                # Connexion à la base de données SQLite
                # print(f"Exécution de la requête SQL pour les coordonnées : x={x_clicked}, y={y_clicked}")
                # Connexion à la base de données SQLite
                conn = sqlite3.connect(chemin_bdd)
                cursor = conn.cursor()

                # Requête SQL pour récupérer le nom du match, le numéro du set, et le numéro du point
                cursor.execute(
                    """
                    SELECT competition, Gamename, numero_set, num_point 
                    FROM Liste_des_coups 
                    WHERE (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?)  
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?)
                """,
                    (
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                    ),
                )
                result = cursor.fetchone()
                conn.close()

                if result is None:
                    print(
                        "Aucune entrée trouvée dans la base de données pour ces coordonnées."
                    )
                    return "", {"display": "none"}

                competition, match_name, set_number, point_number = result
                video_url = f"https://dataroom.liris.cnrs.fr/vizvid/pipeline-tt/{competition}/{match_name}/clips/set_{set_number}_point_{point_number}/set_{set_number}_point_{point_number}.mp4"
                # print(f"URL de la vidéo trouvée : {video_url}")

                return video_url, {
                    "display": "block",
                    "width": "600px",
                    "margin-top": "20px",
                    "margin-left": "20px",
                }
            except Exception as e:
                print(f"Erreur lors de l'exécution du callback : {e}")
                return "", {"display": "none"}

        # Lancement de l'application Dash sur un port libre
        free_port = find_free_port()
        print(
            f"L'application est disponible à l'adresse suivante : http://127.0.0.1:{free_port}"
        )
        app.run(debug=False, port=free_port)

        return None

    def show_global_test_with_video(
        self,
        joueurs: list[str],
        joueur_initial: str,
        effets: list[str],
        num_coups: list[int] = [2],
    ) -> None:
        """
        Effectue l'étude globale réalisée par global_test() sur chaque joueur et affiche la figure encapsulée dans une application Dash permettant d'afficher la vidéo associée à un point de la figure.
        Plusieurs filtres permettent de choisir le joueur, l'effet du coup et le numéro du coup.
        Le test du chi² ainsi que la table de contingence sont affichés également.
        Cliquer sur une cellule de la table de contingence affiche sur la dernière figure les points correspondants à cette case.

        Args:
            joueurs (list): Liste des joueurs disponibles à afficher dans le menu déroulant.
            joueur_initial (str): Le joueur à afficher par défaut dans le menu déroulant.
            effets (list): Liste des effets à afficher dans le menu déroulant.
            num_coups (list, optionnel): Liste des numéros de coups à afficher (uniquement 2 par défaut pour afficher les retours).

        Returns:
            None
        """
        # Création de l'application ----------------------------

        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Récupération des logos ----------------------------

        @app.server.route("/images/<path:image_name>")
        def servir_image(image_name):
            return send_from_directory(chemin_dossier_images, image_name)

        # Initialisation des données ----------------------------

        assert effets is not None
        effet_initial = "Tout"  # Choix par défaut
        effets.insert(0, effet_initial)

        assert num_coups is not None
        if not num_coups:  # On s'assure que la liste est non vide
            num_coups.insert(0, 2)
        num_coup_initial = num_coups[0]

        df_empty = pd.DataFrame(
            columns=[]
        )  # Dataframe qui contiendra les données de la table de contingence

        studies = [
            "Retour_service",
            "Retour_vainqueur",
            "Retour_domination",
            "Retour_pression",
        ]
        study_inital = "Retour_service"

        # ----------------------------

        # Header de l'application ----------------------------
        app.index_string = """
            <!DOCTYPE html>
            <html>
                <head>
                    {%metas%}
                    <title>{%title%}</title>
                    {%favicon%}
                    {%css%}
                    <style>
                        body {
                            background: linear-gradient(white, lightgray);  /* Applique le dégradé au body */
                            margin: 0;
                            padding: 0;
                            min-height: 260vh;  /* Assure que le body couvre toute la hauteur de la page (à ajuster) */
                            overflow-x: hidden;
                        }
                    </style>
                </head>
                <body>
                    {%app_entry%}
                    <footer>
                        {%config%}
                        {%scripts%}
                        {%renderer%}
                    </footer>
                </body>
            </html>
        """

        # Création de la mise en page ----------------------------
        app.layout = html.Div(
            children=[
                dcc.Store(
                    id="store-data", data={}
                ),  # Permettra de stocker temporairement des données
                # Conteneur pour les logos ----------------------------
                html.Div(
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "align-items": "center",
                        "padding": "0px",
                        "justify-content": "flex-start",
                    },
                    children=[
                        html.Img(
                            src="/images/centrale_lyon_logo.png",
                            style={"width": "200px", "margin-left": "30px"},
                            draggable=False,
                        ),
                        html.Img(
                            src="/images/liris_logo.png",
                            style={"width": "200px", "margin-left": "20px"},
                            draggable=False,
                        ),
                    ],
                ),
                # Conteneur pour le titre ----------------------------
                html.Div(
                    style={
                        "display": "flex",
                        "justify-content": "center",
                        "padding": "0px",
                        "line-height": "1",
                    },
                    children=[
                        html.H1(
                            "Analyse des Retours au Tennis de Table",
                            style={
                                "textAlign": "center",
                                "font-size": "2.5em",
                                "margin": "0",
                                "line-height": "1",
                            },
                        )
                    ],
                ),
                # Conteneur pour les dropdowns ----------------------------
                html.Div(
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "center",
                        "align-items": "center",
                        "gap": "10px",
                        "margin-bottom": "20px",
                        "margin-top": "20px",
                    },
                    children=[
                        # Choix du joueur ----------------------------
                        dcc.Dropdown(
                            id="joueur-dropdown",
                            options=[
                                {"label": joueur, "value": joueur}
                                for joueur in joueurs
                            ],
                            value=joueur_initial,
                            style={"width": "300px"},
                        ),
                        # Choix de l'effet ----------------------------
                        dcc.Dropdown(
                            id="effet-dropdown",
                            options=[
                                {"label": effet, "value": effet}
                                for effet in effets
                            ],
                            value=effet_initial,
                            style={"width": "300px"},
                        ),
                        # Choix du numéro du coup ----------------------------
                        dcc.Dropdown(
                            id="num_coup-dropdown",
                            options=[
                                {"label": f"Coup numéro {num}", "value": num}
                                for num in num_coups
                            ],
                            value=num_coup_initial,
                            style={"width": "300px"},
                        ),
                    ],
                ),
                # Conteneur pour la figure et la vidéo ----------------------------
                html.Div(
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "space-around",
                        "padding": "20px",
                    },
                    children=[
                        html.Div(
                            children=[
                                # Affichage de la figure ----------------------------
                                dcc.Graph(
                                    id="pingpong-table",
                                    style={
                                        "height": "500px",
                                        "width": "600px",
                                        "margin-right": "20px",
                                    },
                                )
                            ],
                            style={"flex": "1", "margin-left": "30px"},
                        ),
                        html.Div(
                            id="video-container",
                            children=[
                                # Affichage de la vidéo ----------------------------
                                html.Video(
                                    id="video-player",
                                    controls=True,
                                    autoPlay=True,
                                    src="",
                                    style={
                                        "width": "650px",
                                        "max-width": "100%",
                                        "margin-left": "20px",
                                    },
                                )
                            ],
                            style={"flex": "1", "margin-right": "30px"},
                        ),
                    ],
                ),
                # Conteneur pour afficher le test du chi^2 ----------------------------
                html.Div(
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "left",
                        "align-items": "flex-start",
                        "margin-top": "150px",
                        "margin-left": "10px",
                    },
                    children=[
                        # Affichage de l'intitulé ----------------------------
                        dcc.Markdown(
                            "**Test du $$\\chi^2$$**",
                            mathjax=True,
                            style={"width": "20%", "font-size": "20px"},
                        ),
                        # Affichage du texte contenant les résultats ----------------------------
                        dcc.Markdown(
                            id="texte-explicatif",
                            style={"font-size": "20px"},
                            mathjax=True,
                        ),
                    ],
                ),
                # Ajout de la table de contingence ----------------------------
                html.Div(
                    style={
                        "margin-left": "10px",
                        "margin-bottom": "10px",
                    },
                    children=[
                        # Affichage du titre du tableau ----------------------------
                        html.H3(
                            "Table de contingence", style={"font-size": "20px"}
                        ),
                        # Choix de la table de contingence ----------------------------
                        dcc.Dropdown(
                            id="contingence-dropdown",
                            options=[
                                {"label": study, "value": study}
                                for study in studies
                            ],
                            value=study_inital,
                            style={"width": "300px"},
                        ),
                        # Affichage de la table de contingence  ----------------------------
                        dash_table.DataTable(
                            id="table",
                            columns=[
                                {"name": i, "id": i} for i in df_empty.columns
                            ],
                            data=df_empty.to_dict("records"),
                            style_table={
                                "width": "50%",
                                "margin-left": "auto",
                                "margin-right": "auto",
                            },
                            style_cell={
                                "textAlign": "left",
                                "padding": "10px",
                            },
                            style_header={
                                "backgroundColor": "lightgray",
                                "fontWeight": "bold",
                            },
                        ),
                    ],
                ),
                # Ajout de la figure et de la vidéo associés à la table de contingence ----------------------------
                html.Div(
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "justify-content": "space-around",
                        "padding": "20px",
                    },
                    children=[
                        # Affichage de la figure ----------------------------
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="contingence-data-display",
                                    style={
                                        "height": "500px",
                                        "width": "600px",
                                        "margin-right": "20px",
                                    },
                                )
                            ],
                            style={"flex": "1", "margin-left": "30px"},
                        ),
                        # Affichage de la vidéo ----------------------------
                        html.Div(
                            id="contingence-video-container",
                            children=[
                                html.Video(
                                    id="contingence-video-player",
                                    controls=True,
                                    autoPlay=True,
                                    src="",
                                    style={
                                        "width": "650px",
                                        "max-width": "100%",
                                        "margin-left": "20px",
                                    },
                                )
                            ],
                            style={"flex": "1", "margin-right": "30px"},
                        ),
                    ],
                ),
            ]
        )

        # Mise en place des callback  ----------------------------

        # Callback pour mettre à jour les informations affichées en fonction de la cellule cliquée ----------------------------
        @app.callback(
            Output("contingence-data-display", "figure"),
            Input("table", "active_cell"),  # Cellule cliquée
            Input("table", "data"),  # Données de la table
            Input("store-data", "data"),  # Données stockées
        )
        def update_cell_info(
            active_cell: dict, table_values: list[dict], data: list
        ) -> "Figure":
            from modules.data_interpretation import get_data_from_table

            # Initialisation des variables
            figure_list = [
                (Figure(False), (0, 0))
            ]  # [ (Figure, (rows_number, columns_number)) ]
            rows_number = 0  # Numéro de ligne
            columns_number = 0  # Numéro de colonne
            rows_number_id = -1
            columns_number_id = -1

            if active_cell:
                # Récupère les informations de la cellule cliquée
                rows_number = active_cell["row"]  # Abscisse de la case cliquée
                columns_number = active_cell[
                    "column_id"
                ]  # Ordonnée de la case cliquée

                # Vérifie si la cellule cliquée contient une valeur strictement positive
                if table_values[rows_number][str(columns_number)] > 0:
                    # Extraction des données nécessaires à partir de data
                    (
                        rows,
                        columns,
                        rows_id,
                        columns_id,
                        rows_coords,
                        columns_coords,
                    ) = data[0]
                    row_name = data[1]

                    # figure_list = get_data_from_table(rows, columns, rows_id, columns_id, rows_label, columns_label) # Avec get_data_from_table_old
                    figure_list = get_data_from_table(
                        rows, columns, rows_coords, columns_coords
                    )

                    rows_number_id, columns_number_id = table_values[
                        rows_number
                    ][row_name], eval(
                        list(table_values[0].keys())[columns_number]
                    )  # Nom de la case cliquée (nom abscisse, nom ordonnée)

            if rows_number_id != -1:
                for i in range(
                    len(figure_list)
                ):  # On trouve la bonne figure selon la case cliquée

                    if figure_list[i][1] == (
                        rows_number_id,
                        columns_number_id,
                    ):
                        return figure_list[i][0].get_fig()

            return Figure(
                False
            ).get_fig()  # Si aucune cellule n'est cliquée ou si la valeur dans la cellule est nulle

        # Callback pour mettre à jour la figure, le test du chi2 et la table de contingence en fonction des filtres ----------------------------
        @app.callback(
            [
                Output("pingpong-table", "figure"),
                Output("texte-explicatif", "children"),
                Output("table", "data"),
                Output("table", "columns"),
                Output("store-data", "data"),
            ],  # Mise à jour des colonnes et des données
            [
                Input("joueur-dropdown", "value"),
                Input("effet-dropdown", "value"),
                Input("num_coup-dropdown", "value"),
                Input("contingence-dropdown", "value"),
            ],
        )
        def update_figure(
            joueur_selectionne: str,
            effet_selectionne: str,
            num_coup_selectionne: str,
            study_selectionnee: str,
        ):
            from modules.data_interpretation import global_test

            # Initialisation et récupération des données
            new_figure = Figure(False)
            resultats, data = global_test(
                [new_figure],
                [joueur_selectionne],
                study_selectionnee,
                [effet_selectionne],
                [[num_coup_selectionne]],
            )

            table_contingence = pd.DataFrame(columns=[])
            text = "$$\\textnormal{{Pas de resultats.}}$$"
            data_table = df_empty.to_dict("records")
            columns_table = [{"name": i, "id": i} for i in df_empty.columns]
            index_name = ""
            col_name = ""

            if resultats:
                table_contingence, chi2 = resultats[joueur_selectionne][0]
                if chi2:
                    chi2_value = chi2[0]
                    p_value = chi2[1]
                    ddl = chi2[2]
                    text = f"Statistique du $$\\chi^2 = {chi2_value:.4f}$$, &nbsp;&nbsp;&nbsp; $$p$$-value $$= {p_value:.7f}$$, &nbsp;&nbsp;&nbsp; Degrés de liberté $$= {ddl}$$"

                    index_name = table_contingence.index.name or "Service"
                    col_name = table_contingence.columns.name or "Retour"
                    top_left_label = f"{index_name} \\ {col_name}"
                    table_contingence = table_contingence.reset_index().rename(
                        columns={"index": index_name}
                    )
                    columns_table = [
                        {"name": top_left_label, "id": index_name}
                    ] + [
                        {"name": col, "id": col}
                        for col in table_contingence.columns[1:]
                    ]
                    data_table = table_contingence.to_dict("records")
            else:
                text = "$$\\textnormal{{Pas de resultats.}}$$"
                data_table = df_empty.to_dict("records")
                columns_table = [
                    {"name": i, "id": i} for i in df_empty.columns
                ]

            data = (
                data,
                index_name,
                col_name,
            )  # Données à stocker pour la mise à jour de la table de contingence

            # new_figure.get_fig().write_image("fig2.svg")

            return new_figure.get_fig(), text, data_table, columns_table, data

        # Callback pour mettre à jour la vidéo en fonction du point cliqué ----------------------------
        @app.callback(
            [Output("video-player", "src"), Output("video-player", "style")],
            [Input("pingpong-table", "clickData")],
        )
        def display_video_on_click(clickData):
            if clickData is None:
                return "", {"display": "none"}

            try:
                # Récupérer les coordonnées du point cliqué
                x_clicked = clickData["points"][0]["x"]
                y_clicked = clickData["points"][0]["y"]
                # print(f"Coordonnées cliquées : x={x_clicked}, y={y_clicked}")

                # Connexion à la base de données SQLite
                # print(f"Exécution de la requête SQL pour les coordonnées : x={x_clicked}, y={y_clicked}")
                # Connexion à la base de données SQLite
                conn = sqlite3.connect(chemin_bdd)
                cursor = conn.cursor()

                # Requête SQL pour récupérer le nom du match, le numéro du set, et le numéro du point
                cursor.execute(
                    """
                    SELECT competition, Gamename, numero_set, num_point 
                    FROM Liste_des_coups 
                    WHERE (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?)  
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?)
                """,
                    (
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                    ),
                )
                result = cursor.fetchone()
                conn.close()

                if result is None:
                    print(
                        "Aucune entrée trouvée dans la base de données pour ces coordonnées."
                    )
                    return "", {"display": "none"}

                competition, match_name, set_number, point_number = result
                video_url = f"https://dataroom.liris.cnrs.fr/vizvid/pipeline-tt/{competition}/{match_name}/clips/set_{set_number}_point_{point_number}/set_{set_number}_point_{point_number}.mp4"
                # print(f"URL de la vidéo trouvée : {video_url}")

                return video_url, {
                    "display": "block",
                    "width": "600px",
                    "margin-top": "20px",
                    "margin-left": "20px",
                }
            except Exception as e:
                print(f"Erreur lors de l'exécution du callback : {e}")
                return "", {"display": "none"}

        # Callback pour mettre à jour la vidéo associée à la table de contingence en fonction du point cliqué ----------------------------
        @app.callback(
            [
                Output("contingence-video-player", "src"),
                Output("contingence-video-player", "style"),
            ],
            [Input("contingence-data-display", "clickData")],
        )
        def display_video_on_click_contingence(clickData):
            if clickData is None:
                return "", {"display": "none"}

            try:
                # Récupérer les coordonnées du point cliqué
                x_clicked = clickData["points"][0]["x"]
                y_clicked = clickData["points"][0]["y"]
                # print(f"Coordonnées cliquées : x={x_clicked}, y={y_clicked}")

                # Connexion à la base de données SQLite
                # print(f"Exécution de la requête SQL pour les coordonnées : x={x_clicked}, y={y_clicked}")
                # Connexion à la base de données SQLite
                conn = sqlite3.connect(chemin_bdd)
                cursor = conn.cursor()

                # Requête SQL pour récupérer le nom du match, le numéro du set, et le numéro du point
                cursor.execute(
                    """
                    SELECT competition, Gamename, numero_set, num_point 
                    FROM Liste_des_coups 
                    WHERE (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (coor_balle_x=? AND coor_balle_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?) 
                    OR (pos_joueur_0_x=? AND pos_joueur_0_y=?)  
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?) 
                    OR (pos_joueur_1_x=? AND pos_joueur_1_y=?)
                """,
                    (
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                        x_clicked,
                        y_clicked,
                        -x_clicked,
                        y_clicked,
                        x_clicked,
                        -y_clicked,
                        -x_clicked,
                        -y_clicked,
                    ),
                )
                result = cursor.fetchone()
                conn.close()

                if result is None:
                    print(
                        "Aucune entrée trouvée dans la base de données pour ces coordonnées."
                    )
                    return "", {"display": "none"}

                competition, match_name, set_number, point_number = result
                video_url = f"https://dataroom.liris.cnrs.fr/vizvid/pipeline-tt/{competition}/{match_name}/clips/set_{set_number}_point_{point_number}/set_{set_number}_point_{point_number}.mp4"
                # print(f"URL de la vidéo trouvée : {video_url}")

                return video_url, {
                    "display": "block",
                    "width": "600px",
                    "margin-top": "20px",
                    "margin-left": "20px",
                }
            except Exception as e:
                print(f"Erreur lors de l'exécution du callback : {e}")
                return "", {"display": "none"}

        # Lancement de l'application Dash sur un port libre
        free_port = find_free_port()
        print(
            f"L'application est disponible à l'adresse suivante : http://127.0.0.1:{free_port}"
        )
        app.run(debug=False, port=free_port)

    def get_rebonds(
        self, nom_joueur: str, num_coup: list[int], match: str, effet: str = ""
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[float, float] | None],
        list[tuple[str, str]],
        list[tuple[str, str]],
    ]:
        """
        Renvoie la liste des positions de rebonds pour le joueur donné, le coup et le match donné, ainsi que les coups précédents correspondants.
        Afin de garder la correspondance des indices entre la liste des rebonds et la liste des rebonds précédents, on placera la valeur None si la donnée du rebond précédent n'est pas disponible.
        Renvoie également les id de chacun des rebonds sous forme de couples (id_match, id_coup) pour assurer l'unicité.

        Args:
            nom_joueur (str): Nom du joueur.
            num_coup (list): Numéro du coup.
            match (str): Nom du match.
            effet (str, optionnel): Nom de l'effet.

        Returns:
            tuple: (liste_rebonds, liste_prece, liste_rebonds_id, liste_prece_id)
        """

        conn = sqlite3.connect(chemin_bdd)
        if effet == "Tout" or effet == "":  # Aucun filtre sur l'effet
            try:
                requete_services = f"""
                    SELECT * 
                    FROM Liste_des_coups 
                    WHERE num_coup = {num_coup[0]} 
                    AND joueur_frappe='{nom_joueur}'
                    AND Gamename = '{match}'
                """
                df_rebonds = pd.read_sql_query(requete_services, conn)
            finally:
                conn.close()
        else:  # Filtre sur l'effet
            try:
                requete_services = f"""
                    SELECT * 
                    FROM Liste_des_coups 
                    WHERE num_coup = {num_coup[0]} 
                    AND joueur_frappe='{nom_joueur}'
                    AND Gamename = '{match}'
                    AND effet_coup = '{effet}'
                """
                df_rebonds = pd.read_sql_query(requete_services, conn)
            finally:
                conn.close()

        liste_rebonds = [
            (df_rebonds["coor_balle_x"][i], df_rebonds["coor_balle_y"][i])
            for i in range(len(df_rebonds["coor_balle_x"]))
        ]
        liste_rebonds_id = [
            (df_rebonds["IdMatch"][i], df_rebonds["IdCoup"][i])
            for i in range(len(df_rebonds["coor_balle_x"]))
            if not math.isnan(df_rebonds["coor_balle_x"][i])
        ]
        liste_prece = [
            (
                (
                    df_rebonds["pos_balle_x_prece"][i],
                    df_rebonds["pos_balle_y_prece"][i],
                )
                if pd.notnull(df_rebonds["pos_balle_x_prece"][i])
                else None
            )
            for i in range(len(df_rebonds["pos_balle_x_prece"]))
        ]

        # Calcul de l'id des rebonds précédent
        liste_prece_id = [
            (liste_rebonds_id[i][0], liste_rebonds_id[i][1] - 1)
            for i in range(len(liste_rebonds_id))
        ]

        return liste_rebonds, liste_prece, liste_rebonds_id, liste_prece_id

    def get_player_positions(
        self, nom_joueur: str, num_coup: int, match: str
    ) -> list[tuple]:
        """
        Renvoie la liste des positions du joueur en argument pour le coup et le match donné.

        Args:
            nom_joueur (str): Nom du joueur.
            num_coup (int): Numéro du coup.
            match (str): Nom du match.

        Returns:
            list: La liste des positions correspondantes.
        """
        conn = sqlite3.connect(chemin_bdd)
        try:
            requete_position_joueur = f"""
                SELECT * 
                FROM Liste_des_coups
                WHERE num_coup = {num_coup} 
                AND joueur_frappe='{nom_joueur}'
                AND Gamename = '{match}'
            """
            df_joueur = pd.read_sql_query(requete_position_joueur, conn)
        finally:
            conn.close()

        # On extrait le numéro du joueur (0 ou 1) en regardant la parité de num_coup.
        if num_coup % 2 == 1:  # Serveur
            liste_positions = [
                (
                    df_joueur["pos_joueur_0_x"][i],
                    df_joueur["pos_joueur_0_y"][i],
                )
                for i in range(len(df_joueur["pos_joueur_0_x"]))
            ]
        else:  # Retourneur
            liste_positions = [
                (
                    df_joueur["pos_joueur_1_x"][i],
                    df_joueur["pos_joueur_1_y"][i],
                )
                for i in range(len(df_joueur["pos_joueur_1_x"]))
            ]

        return liste_positions

    def get_list_match(self, joueur: str = "") -> list[str]:
        """
        Renvoie la liste des matchs de la BDD contre des droitiers ou pour un joueur donné.

        Args:
            joueur (str, optionnel): Nom du joueur. Si vide, retourne tous les matchs contre droitiers.

        Returns:
            list: Liste des noms de matchs.
        """
        if joueur == "":
            conn = sqlite3.connect(chemin_bdd)
            try:
                requete_match = """
                    SELECT DISTINCT Gamename
                    FROM Liste_des_coups
                    WHERE joueur_frappe NOT IN ('KRISTIAN-KARLSSON', 'WANG-CHUQIN','IVOR-BAN')
                    AND joueur_sur NOT IN ('KRISTIAN-KARLSSON', 'WANG-CHUQIN','IVOR-BAN')
                """
                df_match = pd.read_sql_query(requete_match, conn)
            finally:
                conn.close()

            return list(df_match["Gamename"])

        else:
            conn = sqlite3.connect(chemin_bdd)
            try:
                requete_match = f"""
                    SELECT DISTINCT Gamename
                    FROM Liste_des_coups
                    WHERE joueur_frappe = '{joueur}'
                    AND joueur_sur NOT IN ('KRISTIAN-KARLSSON', 'WANG-CHUQIN','IVOR-BAN')
                """
                df_match = pd.read_sql_query(requete_match, conn)
            finally:
                conn.close()

            return list(df_match["Gamename"])

    def get_list_joueurs(self) -> list[str]:
        """
        Renvoie la liste des joueurs de la BDD.

        Returns:
            list: Liste des noms des joueurs.
        """
        conn = sqlite3.connect(chemin_bdd)
        try:
            requete_joueur = """
                SELECT DISTINCT nom
                FROM Liste_des_coups
            """
            df_joueur = pd.read_sql_query(requete_joueur, conn)
        finally:
            conn.close()

        return list(df_joueur["nom"])

    def get_effet_joueur(
        self, nom_joueur: str, num_coup: int, match: str
    ) -> list[str]:
        """
        Renvoie la liste des effets des num_coup-ième coup du joueur nom_joueur lors du match match.

        Args:
            nom_joueur (str): Nom du joueur.
            num_coup (int): Numéro du coup.
            match (str): Nom du match.

        Returns:
            list: Liste des effets pour chaque coup.
        """
        conn = sqlite3.connect(chemin_bdd)
        assert (
            num_coup > 1
        )  # Les effets ne sont pas disponibles sur les services
        try:
            requete_effet = f"""
                SELECT * 
                FROM Liste_des_coups
                WHERE num_coup = {num_coup} 
                AND joueur_frappe='{nom_joueur}'
                AND Gamename = '{match}'
            """
            df_effet = pd.read_sql_query(requete_effet, conn)
        finally:
            conn.close()

        liste_effet = [
            df_effet["effet_coup"][i]
            for i in range(len(df_effet["effet_coup"]))
        ]
        return liste_effet

    def get_lateralite_joueur(self, nom_joueur: str, match: str) -> list[str]:
        """
        Renvoie la liste des latéralités des services du joueur nom_joueur lors du match match.

        Args:
            nom_joueur (str): Nom du joueur.
            match (str): Nom du match.

        Returns:
            list: Liste des latéralités correspondantes.
        """
        conn = sqlite3.connect(chemin_bdd)
        try:
            requete_lateralite = f"""
                SELECT * 
                FROM Liste_des_coups
                WHERE num_coup = 1 
                AND joueur_frappe='{nom_joueur}'
                AND Gamename = '{match}'
            """
            df_lateralite = pd.read_sql_query(requete_lateralite, conn)
        finally:
            conn.close()

        liste_lateralite = [
            df_lateralite["service_lateralite"][i]
            for i in range(len(df_lateralite["service_lateralite"]))
        ]
        return liste_lateralite

    def get_winners(
        self, nom_joueur: str, match: str, num_coup: int = 2
    ) -> list[str]:
        """
        Renvoie la liste des gagnants des points du match.
        Le nom du joueur et le numéro du coup servent à éviter de compter plusieurs fois un point en parcourant les échanges d'un même point.

        Args:
            nom_joueur (str): Nom du joueur.
            match (str): Nom du match.
            num_coup (int, optionnel): Numéro du coup qu'on fixe pour ne pas compter plusieurs fois un même point.

        Returns:
            list: Liste des gagnants de chaque point du match.
        """
        conn = sqlite3.connect(chemin_bdd)
        try:
            requete_winner = f"""
                SELECT * 
                FROM Liste_des_coups
                WHERE num_coup = {num_coup} 
                AND joueur_frappe='{nom_joueur}'
                AND Gamename = '{match}'
            """
            df_winner = pd.read_sql_query(requete_winner, conn)
        finally:
            conn.close()

        winner = [
            df_winner["winner"][i]
            for i in range(len(df_winner["coor_balle_x"]))
            if not math.isnan(df_winner["coor_balle_x"][i])
        ]

        return winner

    def get_list_effets(self) -> list[str]:
        """
        Renvoie la liste des effets de la BDD.

        Returns:
            list: Liste des effets.
        """
        conn = sqlite3.connect(chemin_bdd)
        try:
            requete_effets = """
                SELECT DISTINCT effet_coup
                FROM Liste_des_coups
            """
            df_effets = pd.read_sql_query(requete_effets, conn)
        finally:
            conn.close()

        return list(df_effets["effet_coup"])

    def get_list_num_coups(self) -> list[int]:
        """
        Renvoie la liste des numéros de coup de la BDD.

        Returns:
            list: Liste des numéros de coup.
        """
        conn = sqlite3.connect(chemin_bdd)
        try:
            requete_coup = """
                SELECT DISTINCT num_coup
                FROM Liste_des_coups
            """
            df_coup = pd.read_sql_query(requete_coup, conn)
        finally:
            conn.close()

        return list(df_coup["num_coup"])
