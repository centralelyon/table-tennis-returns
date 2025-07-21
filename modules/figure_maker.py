import numpy as np
import plotly.graph_objects as go  # Pour générer des graphiques interactifs

from modules.data_processing import place_on_side

# Couleurs utilisées notamment pour colorer les clusters
couleurs = [
    "#1f77b4",  # Bleu
    "#ff7f0e",  # Orange
    "#2ca02c",  # Vert
    "#d62728",  # Rouge
    "#9467bd",  # Violet
    "#8c564b",  # Marron
    "#e377c2",  # Rose
    "#7f7f7f",  # Gris
    "#bcbd22",  # Or
    "#17becf",  # Aigue-marine
]

longueur = 274  # longueur de la table en centimètres
largeur = 152  # largeur de la table en centimètres


def draw_table(draw_grid=False) -> go.Figure:
    """
    Dessine une table interactive, avec éventuellement une grille divisant chaque demie-table en neuf cases égales.

    Args:
        draw_grid (bool): Indique si on veut tracer un quadrillage coupant les demies-tables en 9.

    Returns:
        go.Figure: Objet Plotly représentant la table.
    """

    # Création de la figure Plotly
    fig = go.Figure()

    # Ajout de la surface de la table
    fig.add_trace(
        go.Scatter(
            x=[
                -largeur / 2,
                largeur / 2,
                largeur / 2,
                -largeur / 2,
                -largeur / 2,
            ],
            y=[
                -longueur / 2,
                -longueur / 2,
                longueur / 2,
                longueur / 2,
                -longueur / 2,
            ],
            fill="toself",
            fillcolor="#add8e6",
            line=dict(color="black", width=2),
            mode="lines",
            showlegend=False,
        )
    )

    # Ajout de la ligne médiane
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-longueur / 2, longueur / 2],
            line=dict(color="#d1f2f8", width=3),
            mode="lines",
            showlegend=False,
        )
    )

    # Ajout du filet
    fig.add_trace(
        go.Scatter(
            x=[-largeur / 2, largeur / 2],
            y=[0, 0],
            line=dict(color="black", width=8),
            mode="lines",
            showlegend=False,
        )
    )

    # Option pour ajouter un quadrillage
    if draw_grid:
        # Lignes verticales
        for x in [-largeur / 6, largeur / 6]:
            fig.add_trace(
                go.Scatter(
                    x=[x, x],
                    y=[-longueur / 2, longueur / 2],
                    line=dict(color="grey", width=1, dash="dash"),
                    mode="lines",
                    showlegend=False,
                )
            )

        # Lignes horizontales
        for y in [-longueur / 3, -longueur / 6, longueur / 6, longueur / 3]:
            fig.add_trace(
                go.Scatter(
                    x=[-largeur / 2, largeur / 2],
                    y=[y, y],
                    line=dict(color="grey", width=1, dash="dash"),
                    mode="lines",
                    showlegend=False,
                )
            )

    # Configuration supplémentaire pour la visualisation, notamment pour respecter les proportions de la table
    fig.update_layout(
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)",  # Couleur de fond du graphique
        paper_bgcolor="rgba(0,0,0,0)",  # Fond du papier (extérieur) transparent
        showlegend=False,
        width=320,
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def draw_joueurs(fig: go.Figure, positions_joueur: list, joueur: int) -> None:
    """
    Affiche les positions de joueur stockées dans positions_joueur en tenant compte du côté grâce à joueur.

    Args:
        fig (go.Figure): Figure sur laquelle afficher les positions de joueur.
        positions_joueur (list): Liste de coordonnées des positions de joueur à afficher.
        joueur (int): Numéro du joueur auquel correspondent les positions (0 pour le serveur, 1 pour le retourneur).

    Returns:
        None
    """
    if positions_joueur:
        if joueur == 0:  # Serveur
            positions_joueur, [], [], [] = place_on_side(
                positions_joueur, [], [], []
            )
        if joueur == 1:  # Retourneur
            [], [], positions_joueur, [] = place_on_side(
                [], [], positions_joueur, []
            )
        positions_joueur_x = [
            -positions_joueur[i][0] for i in range(len(positions_joueur))
        ]
        positions_joueur_y = [
            positions_joueur[i][1] for i in range(len(positions_joueur))
        ]
        fig.add_trace(
            go.Scatter(
                x=positions_joueur_x,
                y=positions_joueur_y,
                mode="markers",
                marker=dict(color="black", symbol="x", size=7),
                name="Positions des joueurs",
            )
        )
    return


def draw_rebonds(
    fig: go.Figure, positions_rebonds: list, joueur: int, colors=[]
) -> None:
    """
    Affiche les rebonds stockés dans positions_rebonds en tenant compte du côté grâce à joueur.

    Args:
        fig (go.Figure): Figure sur laquelle afficher les rebonds.
        positions_rebonds (list): Liste de coordonnées des rebonds à afficher.
        joueur (int): Numéro du joueur auquel correspondent les rebonds (0 pour le serveur, 1 pour le retourneur).
        colors (list, optionnel): Liste de couleurs pour chaque rebond.

    Returns:
        None
    """
    if positions_rebonds:
        if joueur == 0:  # Serveur
            [], positions_rebonds, [], [] = place_on_side(
                [], positions_rebonds, [], []
            )
        if joueur == 1:  # Retourneur
            [], [], [], positions_rebonds = place_on_side(
                [], [], [], positions_rebonds
            )
        positions_rebonds_x = [
            (
                -positions_rebonds[i][0]
                if positions_rebonds[i] is not None
                and positions_rebonds[i][0] is not None
                else None
            )
            for i in range(len(positions_rebonds))
        ]
        positions_rebonds_y = [
            (
                positions_rebonds[i][1]
                if positions_rebonds[i] is not None
                and positions_rebonds[i][1] is not None
                else None
            )
            for i in range(len(positions_rebonds))
        ]
        if colors != []:
            fig.add_trace(
                go.Scatter(
                    x=positions_rebonds_x,
                    y=positions_rebonds_y,
                    mode="markers",
                    marker=dict(color=colors, size=10),
                    name="Rebonds",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=positions_rebonds_x,
                    y=positions_rebonds_y,
                    mode="markers",
                    marker=dict(color="yellow", size=10),
                    name="Rebonds",
                )
            )

    return


def draw_clusters(
    fig: go.Figure,
    enveloppes: list,
    labels: list,
    positions_rebonds: list,
    k: int,
) -> None:
    """
    Affiche les clusters correspondant aux enveloppes, labels, et rebonds en argument. k est utilisé dans l'algorithme des k moyennes.

    Args:
        fig (go.Figure): Figure sur laquelle afficher les clusters.
        enveloppes (list): Liste contenant les différentes enveloppes.
        labels (list): Liste des labels de chaque point de positions_rebonds.
        positions_rebonds (list): Liste des coordonnées des rebonds pour lesquels afficher les clusters.
        k (int): Nombre de clusters utilisé dans l'algorithme des k moyennes.

    Returns:
        None
    """
    positions_rebonds = [
        (-positions_rebonds[i][0], positions_rebonds[i][1])
        for i in range(len(positions_rebonds))
    ]
    for i in range(k):
        points_du_cluster = np.array(positions_rebonds)[labels == i]
        if enveloppes:
            if enveloppes[i] is not None:
                hull = enveloppes[i]
                for simplex in hull.simplices:
                    fig.add_trace(
                        go.Scatter(
                            x=[
                                points_du_cluster[simplex, 0][0],
                                points_du_cluster[simplex, 0][1],
                            ],
                            y=[
                                points_du_cluster[simplex, 1][0],
                                points_du_cluster[simplex, 1][1],
                            ],
                            mode="lines",
                            line=dict(color=couleurs[i]),
                            showlegend=False,
                        )
                    )
    return
