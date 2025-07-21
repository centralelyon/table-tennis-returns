# Analysis of Table Tennis Service Returns

## Project Overview

This project aims to analyze table tennis match data, focusing on **service returns**. The goal is to group these returns into **clusters** using machine learning methods, then interpret these groupings to better understand player strategies and behaviors depending on the type of serve received.

The analysis is based on a database provided by LIRIS, containing in particular:

- The coordinates of bounces,
- The hitting positions,
- The clusters of bounces,
- And many other details from numerous professional matches.

---

## Main Features

### 1. Data Display and Exploration

- **Notebook `1_display_data.ipynb`**  
  Visual exploration of table tennis match data:
    - Interactive and static visualization of bounces and shots on the table,
    - Display of isolated point videos,
    - Exploration of bounce clusters,
    - Use of Plotly figures for analyzing positions and trajectories.

### 2. Calculation and Display of Domination and Pressure Indicators

- **Notebook `2_domination_pression.ipynb`**  
  Calculation and visualization of two advanced metrics:
    - **Domination**: a dynamic indicator taking into account the score, set difference, winner of the previous point, and match evolution,
    - **Pressure**: an indicator measuring the pressure felt at each point (score gap, key moments, end of set, decisive set),
    - Graphical display of the evolution of these indicators throughout the match.

### 3. Robustness and Convergence of Results

- **Notebook `3_gather_results.ipynb`**  
  Repeat chi-squared tests many times to eliminate the effect of random cluster initialization (k-means).  
  Analyze the convergence of p-values and the robustness of the results.

### 4. Descriptive Statistics on Returns

- **Notebook `4_statistiques.ipynb`**  
  Calculate and visualize:
    - The percentages of flips, topspins, pushes, and blocks on return,
    - The attack/defense/intermediate distributions,
    - The number of data points available per player.

### 5. Multi-Match Comparative Analysis

- **Notebook `5_matchs_lebrun.ipynb`**  
  Compare Alexis Lebrun's return strategies across several matches, studying the dependence between the received serve zone and the chosen return zone.

### 6. Dependence Between Return and Set Number

- **Notebook `6_dependance_num_set.ipynb`**  
  Analyze whether the return zone varies according to the set number, for different players.

### 7. Study of the Third Shot After the Return

- **Notebook `7_etude_3eme_coup.ipynb`**  
  Analyze the relationship between the chosen return zone and the success of the third shot (point won by the opponent).

---

## Dash Application

A Dash application has been developed to facilitate data exploration.  
Bounces are displayed on a Plotly figure:

- Clicking on a point displays the video of the corresponding rally,
- Several filters allow you to select a player, handedness, or shot number.

<p align="center">
   <img src="images/dash_app.png" alt="dash app example" width="1100">
</p>

---

## Advanced Analysis: Domination and Pressure

### Domination Indicator

An advanced indicator has been developed to quantify **domination** during a match, taking into account:

- The score,
- The set difference,
- The winner of the previous point,
- The dynamic evolution of the match.

<p align="center">
  <img src="images/domination_with_cluster.png" alt="Domination with clusters" width="600"/>
</p>

### Pressure Indicator

Another indicator measures the **pressure** felt by a player at each point, combining:

- The score gap,
- Key moments (set point, match point),
- End of set,
- Decisive set.

<p align="center">
  <img src="images/pression_with_cluster.png" alt="Pressure with clusters" width="600"/>
</p>

---

## Installation

1. Clone this GitHub repository:

   ```bash
   git clone https://github.com/centralelyon/tt-returns.git
   cd tt-returns
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Authors

- Riad ATTOU : <attou.rd@gmail.com>
- Marin MATHE : <mathe.marin22@gmail.com>
- Aymeric ERADES : <aymeric.erades@ec-lyon.fr>
- Romain VUILLEMOT : <romain.vuillemot@gmail.com>
