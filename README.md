# Optimisation par Descente de Gradient Stochastique (SGD)

## Problème d'optimisation

Nous cherchons à minimiser une fonction de coût définie comme une espérance :

$$
f(x) = \mathbb{E}[F(x, \xi)]
$$

où la fonction cible est donnée par :

$$
F(x, \xi) = (x - \xi)^2 + \sin(x - \xi) + e^{-\xi}(x^2 - 2x + 1)
$$

avec $\xi$ suivant une loi normale : $\xi \sim \mathcal{N}(\mu, \sigma^2)$.

La solution théorique $x^*$ est le minimum global de $f(x)$. Nous utilisons l'algorithme **SGD** (Stochastic Gradient Descent) pour approximer cette solution.

## Gradient Stochastique

Le gradient de $F(x, \xi)$ est donné par :

$$
\nabla F(x, \xi) = 2(x - \xi) + \cos(x - \xi) + e^{-\xi}(2x - 2)
$$

À chaque itération $n$, la mise à jour de $x$ suit la règle :

$$
x_{n+1} = x_n - \gamma_n \nabla F(x_n, \xi_n)
$$

avec :
- $\gamma_n$ un pas d'apprentissage,
- $\xi_n$ un échantillon aléatoire indépendant.

Nous choisissons une séquence de pas satisfaisant les conditions de Robbins-Siegmund :

$$
\sum_{n=1}^\infty \gamma_n = \infty, \quad \sum_{n=1}^\infty \gamma_n^2 < \infty
$$

avec :

$$
\gamma_n = \frac{\gamma_0}{1 + n^\alpha}, \quad \alpha \in [0.5, 1]
$$

## Approximations et Visualisations

Nous approximons $f(x)$ par une moyenne empirique :

$$
f(x) \approx \frac{1}{N} \sum_{i=1}^{N} F(x, \xi_i)
$$

et son gradient :

$$
\nabla f(x) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla F(x, \xi_i)
$$

Le zéro du gradient correspond aux points critiques de la fonction.

Nous simulons plusieurs valeurs de $\xi$ pour observer la variance du gradient stochastique et la difficulté d'approximation par SGD :

$$
\text{Var}(\nabla F(x, \xi))
$$

Nous analysons aussi l'influence de $\xi$ sur $F(x, \xi)$ en le fixant à certaines valeurs typiques ($\mu$, $\mu \pm \sigma$, $\mu \pm 2\sigma$).

## Convergence et Analyse Théorique

Le théorème de Robbins-Siegmund garantit que sous certaines conditions :

1. $\|x_{n+1} - x_n\| \to 0$ presque sûrement.
2. La suite $L(x_n)$ converge presque sûrement vers une limite $L_\infty$.

Nous utilisons la fonction de Lyapunov suivante :

$$
L(x_n) = \frac{1}{2} (x_n - x^*)^2
$$

et analysons l'évolution de $L(x_n)$ au cours des itérations.

Nous utilisons également l'expression suivante pour estimer l'erreur totale :

$$
\frac{1}{\Gamma(1 + CL \gamma_k^2)} \left( CL \sum_{k \geq n+2} \gamma_k^2 + \frac{CL \gamma_{n+1}^2}{1 + CL \gamma_{n+1}^2} \right)
$$

avec :
- $CL$ une constante positive,
- Le premier terme mesurant l'impact immédiat du bruit stochastique,
- Le second terme représentant l'effet cumulé des itérations futures.

## Paramètres de Simulation

- **Distribution de $\xi$** : $\mathcal{N}(2,1)$
- **Paramètres du pas** : $\gamma_n = \frac{1}{n^{0.7}}$
- **Nombre d'itérations** : $N_0 = 1000$
- **Constante de Lyapunov** : $CL = 1.0$

## Implémentation

Le projet inclut :
- Une implémentation de **SGD** pour minimiser $f(x)$,
- Des visualisations de la fonction de coût et du gradient,
- Une analyse de la convergence basée sur Robbins-Siegmund,
- Une étude des variations de $F(x, \xi)$ en fonction de $\xi$.

---

🚀 **Ce projet permet de mieux comprendre la convergence de SGD en présence de bruit et de vérifier expérimentalement les conditions théoriques de Robbins-Siegmund.**

