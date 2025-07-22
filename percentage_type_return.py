#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour générer la figure stack_percentage_type_return.png
Barres horizontales empilées des pourcentages de types de retour par joueur,
avec titres et labels en LaTeX, en utilisant vos fonctions de récupération.
TRIÉ PAR TAUX DE TOPSPIN DÉCROISSANT.
"""

import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1) Connexion à la base de données ---
DB_PATH = "BDD_avec_cluster.db"
conn = sqlite3.connect(DB_PATH)


# --- 2) Vos fonctions d'extraction ---
def get_joueurs(conn):
    query = (
        "SELECT DISTINCT joueur_frappe FROM Liste_des_coups WHERE num_coup = 2"
    )
    return [row[0] for row in conn.execute(query).fetchall()]


def get_pourcentages_retour(joueur, conn):
    df = pd.read_sql_query(
        f"""
        SELECT effet_coup
        FROM Liste_des_coups
        WHERE num_coup = {2}
          AND joueur_frappe = '{joueur}'
          AND effet_coup IS NOT NULL
        """,
        conn,
    )
    total = len(df)

    if total == 0:
        return [0, 0, 0, 0]
    types = ["topspin", "flip", "poussette", "bloc"]

    # calcul en pourcentage pour chaque type
    return [100 * (df["effet_coup"] == t).sum() / total for t in types]


# --- 3) Construction du DataFrame ---
joueurs = get_joueurs(conn)
types_orig = ["topspin", "flip", "poussette", "bloc"]
labels_tex = [r"Topspin", r"Flip", r"Push shot", r"Block"]
couleurs = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
]  # rouge, bleu, vert, orange

# on prépare une table vide
data = {t: [] for t in types_orig}
for j in joueurs:
    pct = get_pourcentages_retour(j, conn)
    for i, t in enumerate(types_orig):
        data[t].append(pct[i])
conn.close()

df = pd.DataFrame(data, index=joueurs)

# --- 4) TRI PAR TAUX DE TOPSPIN DÉCROISSANT ---
df_sorted = df.sort_values(by="topspin", ascending=False)

# --- 5) Création de la figure Matplotlib ---
plt.rcParams.update(
    {"font.size": 14, "text.usetex": True, "font.family": "serif"}
)

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(df_sorted))
left = np.zeros(len(df_sorted))

# Utilisation du DataFrame trié
joueurs_tries = df_sorted.index.tolist()

for t, label, color in zip(types_orig, labels_tex, couleurs):
    ax.barh(
        y=y_pos,
        width=df_sorted[t],
        left=left,
        height=0.8,
        label=label,
        color=color,
        edgecolor="white",
    )
    left += df_sorted[t]

# Réglages finaux
ax.set_yticks(y_pos)
ax.set_yticklabels(joueurs_tries)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel(r"Percentage", labelpad=10)
ax.set_title(
    r"Distribution of Return Shot Types by Player (Sorted by Topspin Rate)",
    pad=15,
)
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

leg = ax.legend(loc="lower right", frameon=True)
leg.get_frame().set_edgecolor("black")

plt.tight_layout()
plt.savefig("stack_percentage_type_return.png", dpi=300)
