---
title: "Mind the dual gap"
categories: ML
tags: [ML]
toc: true
math: true
---


## Introduction

On a implémenté dernièrement le solveur classique du LASSO (coordinate descent), néanmoins sur de grosses bases de données et où par exemple on a peu d'individu pour beaucoup d'observations, le solveur est trop lent.  

Les implémentations SOTA (State of the art) du LASSO utilisent de l'optimisation convexe pour accélérer les solveurs. J'implémente et présente ici l'article **Mind the duality gap: safer rules for the Lasso** (RAJOUTER REF) qui utilisent ce qu'on appelle des *screening rules*. En effet, avant d'appliquer le solveur, on va chercher à enlever les features inutiles (donc telles que $\beta_j = 0$ à l'optimum) avant et pendant le solveur.  

L'article utilise astucieusement le duality gap pour créer des règles performantes (utilisé dans sci-kit.learn par exemple) améliorant la rapidité et la convergence du solveur.


Je me suis basé principalement sur le livre de Boyd Convex Optimization pour acquérir les connaissances nécéssaires à implémenter et comprendre cet article

## Un peu de théorie

Avant de comprendre ce que fait l'article et pourquoi ça marche, on va d'abord changer de point de vue sur le LASSO:

Je rappelle que le problème Primal du lasso est:

$$
\hat{\beta}(\lambda)
=
\arg\min_{\beta}
\left(
  \frac{\| y - X\beta \|_2^2}{2}
+
\lambda\| \beta \|_1
\right)
$$ 
où $X \in \mathbb{R}^{n\times p}$ et $\beta \in \mathbb{R}^{p}$

En décomposant entre résidus et features (on pose $z = X\beta - y$ pour le problème primal) et en utilisant le lagrangien:

$$
L(z,\theta, \beta)=\frac{\|z \|_2^2}{2} +\theta^{T}((y-X \beta)-z) + \lambda | \beta \|_1
$$
où on a introduit $\theta$ la variable duale.  
On obtient ainsi le problème dual après avoir minimisé en $z$ et en $\beta$ :

$$
\hat{\theta}(\lambda)
=
\arg\max_{\theta' \in \Delta_X}
\frac{1}{2}\|y\|_2^2
-
\frac{\lambda^2}{2}
\left\|
\theta' - \frac{y}{\lambda}
\right\|_2^2 
=\arg\max_{\theta' \in \Delta_X} g(\theta')
$$
où $\Delta_X = \{\theta' \in \mathbb{R}^n : |x_j^\top \theta'| \le 1,\ \forall j \in \{1,.. .,p\}\}$ est le dual feasible set avec $\theta'=\frac{\theta}{\lambda}$  (la condition provient de la minimisation en $\beta$).  
On observe que le problème dual revient à:
$$
\hat{\theta}(\lambda)
=
\arg\min_{\theta' \in \Delta_X}

\left\|
\theta' - \frac{y}{\lambda}
\right\|_2 
$$
et donc que $\hat{\theta}(\lambda)$ est la projection de $\frac{y}{\lambda}$ sur $\Delta_X$ avec la norme euclidienne. $\Delta_X$ étant convexe et fermé, celle-ci est unique par théorème.  
La solution du problème primal n'est, elle, pas forcément unique (en fonction du rang de X).  

On note aussi que à l'optimum, $\lambda\hat{\theta}(\lambda)+X\hat{\beta}(\lambda)=y$, où l'on a pris en compte la normalisation par rapport à $\lambda$.

**Dans la suite on notera $\theta$ et plus $\theta'$ même s'il s'agit de la version normalisée.**  
En utilisant les conditions de KKT sur $L(z,\theta, \beta)$, on trouve des conditions sur les $\beta_j$ (ce sont les mêmes genre de calculs qui ont été fait dans mon dernier post qui permettent d'aboutir à ce résultat):  

$\hat{\beta}_{j}(\lambda)=0$ dès que $|x_j^{T} \hat{\theta}(\lambda) |<1$

Le problème dual donne une solution unique et des conditions géométriques simples, seul souci: on ne connaît pas la solution duale !

## Screening rules

Notre but ici, c'est d'éliminer correctement les variables inutiles, être une variable $\beta_j$ inutile, ici ça veut dire que $|x_j^{T} \hat{\theta}(\lambda) |<1$.  

L'idée maintenant, c'est que comme on connaît pas la solution duale, on va regarder des ensembles $C$ un peu plus gros qui la **contiennent** (on appelle ces ensembles *safe regions*) et demander une propriété plus forte.  
Si celle-ci est vérifiée, on saura qu'on pourra rejeter la variable $\beta_j$ sans connaître la solution duale.

La propriété va correspondre à notre *safe rule* et ici, il s'agit de:
$$
\mu_C(x_j):= \sup_{\theta \in C} |x_j^{T}\theta| <1
$$

Comme la solution duale est dans l'ensemble $C$, on sait pour sûr que $\beta_j=0$.

On a pas encore défini l'ensemble $C$, mais on voit que si on peut calculer $\mu_C(x_j)$ explicitement et facilement, les calculs seront légers d'un point de vue computationnel.
Pour cela, il faut bien choisir $C$, donc prendre un ensemble simple (pour limiter le coût de calcul de la safe rule) et le plus petit possible pour éliminer le plus grand nombre de variables inutiles.

L'article utilise une boule particulière, mais des dômes ont aussi déjà pu être utilisé ou d'autres figures géométriques.

Si l'on se base sur 1 seul ensemble $C$ que l'on utilise ,disons juste avant le solveur, on a une *règle statique*.  
L'article propose une *règle dynamique* qui agit à chaque étape du solveur ( itératif, coordinate descent) et pour ça, donne une suite d'ensemble qui converge (dans le sens où le diamètre tend vers 0) vers $\{ \hat{\theta (\lambda)} \}$.

## Mind the duality gap

### Les débuts

Pour une boule $B(c,r)$, on a $\mu_C(x_j)=|x_j^{T}c|+r||x_j||$.  
L'article utilise le duality gap pour trouver une suite de boule adéquate.  
En effet, les conditions de Slater étant vérifiées dans le lasso, on a la dualité faible:
$$
\frac{1}{2}\|y\|^2
-
\frac{\lambda^2}{2}
\left\|\theta - \frac{y}{\lambda}\right\|^2
\le
\frac{1}{2}\|X\beta - y\|^2
+
\lambda \|\beta\|_1 
$$
Pour $\theta \in \Delta_X$ !!!  
On extrait donc une minoration de $\|\theta - \frac{y}{\lambda}\|$ pour tout $\theta \in \Delta_X$ et donc en particulier pour $\hat{\theta}(\lambda)$:
$$
\left\lVert \theta - \frac{y}{\lambda} \right\rVert
\ge
\frac{\sqrt{\left(\lVert y \rVert^2 - \lVert X\beta - y \rVert^2 - 2\lambda \lVert \beta \rVert_1\right)_+}}{\lambda}
=\hat{R}_{\lambda}(\beta)
$$

On fixe maintenant un point $\theta \in \Delta_X$, alors $\widecheck{R}_{\lambda}(\theta)= \|\theta - \frac{y}{\lambda}\| \geq \|\hat{\theta}(\lambda) - \frac{y}{\lambda}\|$ par définition de la projection.  
Ainsi la solution duale est dans couronne ou **annulus** :
$$\hat{\theta}(\lambda) \in A(\frac{y}{\lambda},\widecheck{R}_{\lambda}(\theta),\hat{R}_{\lambda}(\beta))$$

### Intuition géométrique 

Le duality gap nous apprend que la solution duale $\hat{\theta}(\lambda)$ est dans une couronne centrée en $\frac{y}{\lambda}$, mais cette information reste trop grossière: une couronne n'est pas une forme géométrique simple et pratique pour notre screening.  
On va exploiter la structure convexe du problème pour trouver notre boule adéquate et j'explique ici comment comment ils y arrivent dans l'article.

Déjà rappelons que $\hat{\theta}(\lambda)$ est la projection de $\frac{y}{\lambda}$ sur $\Delta_X$ qui est **convexe fermé**, donc $[\theta,\hat{\theta}(\lambda)]$ reste dans l'ensemble. Or comme $\hat{\theta}(\lambda)$ est le point de $\Delta_X$ le plus proche de $\frac{y}{\lambda}$, ce segment ne peut pas entrer dans la boule intérieure de la couronne, (borne donnée par le duality gap).  
$\hat{\theta}(\lambda)$ est donc relativement contraint et le point le plus éloigné de $\theta$ parmi les points compatibles avec ces contraintes est obtenu lorsque le segment partant de $\theta$ est tangent à la boule intérieure. Ce point est noté $\theta_{int}$ par l'article et donne notre rayon de sécurité.  
On obtient ainsi une boule centrée en $\theta$ et contenant la solution duale.  
Pour conclure, on obtient comme safe region 
$$
C'=B(\theta,(\tilde{R}_{\lambda}(\theta)^2 - \tilde{R}_{\lambda}(\beta)^2)^{1/2})
$$
où le rayon correspond à $||\theta_{int}-\theta||$ (cf figure 1)



### qui est $\theta$ ? + Converging regions

Je fais un léger point sur le $\theta$ que l'on a fixé plus haut.  
Comme dit précédemment, on va utiliser notre screening rule sur des algorithmes itératifs, par conséquent on va construire $\theta_k$ à partir de $\theta_{k-1}$ et ce sera donc le point précédent qui sera le centre de notre boule à l'étape k.

On note $r'(\theta,\beta)$ le rayon de la boule choisie.  
On a que $r'(\theta,\beta)^2 \leq r(\theta,\beta):= \frac{2}{\lambda^2}G(\theta,\beta)$ où $G(\theta,\beta)$ correspond au duality gap et $r(\hat{\theta}(\lambda),\hat{\beta}(\lambda))=0$ avec la dualité forte.  

On va utiliser le duality gap comme rayon et non le rayon trouvé géométriquement, parce que celui-ci est un critère d'arrêt standard dans les solveurs et est donc déjà calculé. En effet, si $G(\theta,\beta) \leq \epsilon$, alors par dualité forte $P_{\lambda}(\beta)-P_{\lambda}(\hat{\beta}(\lambda)) \leq \epsilon$.     
C'est donc numériquement favorable.  

On note donc $C=B(\theta,r(\theta,\beta))$  
On a construis notre rayon à partir de $G(\theta,\beta)$, donc si le solveur cv vers $(\hat{\theta}(\lambda),\hat{\beta}(\lambda))$, alors le rayon tend vers 0.  
Ainsi $C_{k}=B(\theta_k,r(\theta_k,\beta_k))$ est une safe region qui converge vers $\{\hat{\theta}(\lambda)\}$ quand $\lim(\theta_k,\beta_k)=(\hat{\theta}(\lambda),\hat{\beta}(\lambda))$.  





