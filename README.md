# Optimisation Stochastique avec SGD

## Description du Projet
Ce projet vise à minimiser une fonction de coût définie comme une espérance :

\[
f(x) = \mathbb{E}[F(x, \xi)],
\]

avec la fonction cible :

\[
F(x, \xi) = (x - \xi)^2 + \sin(x - \xi) + e^{-\xi}(x^2 - 2x + 1),
\]

et $\xi$ suit une loi normale $\mathcal{N}(\mu, \sigma^2)$.

Nous utilisons l'algorithme **Stochastic Gradient Descent (SGD)** pour approximer la solution optimale $x^*$ en mettant à jour $x$ à chaque itération selon :

\[
x_{n+1} = x_n - \gamma_n \nabla F(x_n, \xi_n),
\]

avec un pas d'apprentissage adaptatif :

\[
\gamma_n = \frac{\gamma_0}{1 + n^\alpha}, \quad \alpha \in [0.5, 1].
\]

## Objectifs
- Approximer $x^*$ par SGD.
- Analyser la convergence mathématiquement et visuellement.
- Visualiser la variance du gradient stochastique.
- Etudier le comportement de $F(x, \xi)$ en fonction de $x$ et $\xi$.

## Conditions de Convergence
L'algorithme suit le théorème de **Robbins-Siegmund**, qui garantit la convergence sous certaines hypothèses, notamment :

\[
\sum_{n=1}^{\infty} \gamma_n = \infty, \quad \sum_{n=1}^{\infty} \gamma_n^2 < \infty.
\]

Une fonction de Lyapunov $L(x_n) = \frac{1}{2} (x_n - x^*)^2$ est utilisée pour analyser la stabilité et la convergence.

## Implémentation
L'approximation de $f(x)$ et de son gradient est faite par moyennage empirique :

\[
f(x) \approx \frac{1}{N} \sum_{i=1}^{N} F(x, \xi_i),\quad \nabla f(x) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla F(x, \xi_i).
\]

Où les $\xi_i$ sont échantillonnés depuis $\mathcal{N}(\mu, \sigma^2)$.

## Visualisation et Analyse
- **Tracé de $f(x)$** pour observer les minimas locaux et globaux.
- **Visualisation de la variance du gradient stochastique**.
- **Tracé de $L(x_n)$** pour suivre la convergence.
- **Estimation de l'erreur résiduelle** à l'aide de la formule :

\[
\frac{1}{\Gamma(1 + CL \gamma_k^2)} \left( CL \sum_{k \geq n+2} \gamma_k^2 + \frac{CL \gamma_{n+1}^2}{1 + CL \gamma_{n+1}^2} \right).
\]

## Paramètres Utilisés
- $\mu = 2, \sigma = 1$
- $\alpha = 0.7$
- $\gamma_0 = 0.1$
- Nombre d'itérations : $N_0 = 1000$
- CL = 1.0

## Prérequis
- Python 3.x
- Numpy, Matplotlib

## Exécution
Exécutez le script principal :

```bash
python sgd_optimization.py
```

## Auteurs
Projet d’optimisation stochastique utilisant SGD pour l’approximation de minimas d’une fonction de coût probabiliste.

---

Ce fichier **README** est une présentation générale du projet. Pour plus de détails, référez-vous au code source et aux commentaires inclus dans les scripts Python.

