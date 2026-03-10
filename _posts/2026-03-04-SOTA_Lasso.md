---
title: "Mind the dual gap, LASSO"
categories: ML
tags: [ML]
toc: true
math: true
---


## Introduction

Le solveur classique du LASSO par coordinate descent devient vite coûteux lorsque le nombre de variables est très grand devant le nombre d’observations (p≫n).  
Les implémentations modernes accélèrent ce solveur à l’aide de screening rules, qui permettent d’éliminer de manière sûre des variables dont le coefficient optimal est nul.  
Dans ce billet, j’implémente et j’explique l’article [Mind the Duality Gap: Safer Rules for the Lasso](https://arxiv.org/pdf/1505.03410), qui exploite le duality gap pour construire des régions de sécurité convergentes et améliorer fortement le coût du solveur.


Je me suis principalement appuyé sur Convex Optimization de Boyd et Vandenberghe pour acquérir les connaissances nécessaires à implémenter et comprendre cet article

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

En décomposant entre résidus et features (on pose $\rho = y - X \beta$ pour le problème primal) et en utilisant le lagrangien:

$$
L(\rho,\theta, \beta)=\frac{\|\rho \|_2^2}{2} +\theta^{T}((y-X \beta)-\rho) + \lambda | \beta \|_1
$$
où on a introduit $\theta$ la variable duale.  
On obtient ainsi le problème dual après avoir minimisé en $\rho$ et en $\beta$ :

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
En utilisant les conditions de KKT sur $L(\rho,\theta, \beta)$, on trouve des conditions sur les $\beta_j$ (c'est le même genre de calculs qui ont été fait dans mon dernier post qui permettent d'aboutir à ce résultat):  

$$\hat{\beta}_{j}(\lambda)=0\ \text{dès que} \ |x_j^{T} \hat{\theta}(\lambda) |<1$$

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

L'article utilise une boule particulière, mais des dômes ont aussi déjà été utilisés ou d'autres figures géométriques.

Si l'on se base sur 1 seul ensemble $C$ que l'on utilise ,disons juste avant le solveur, on a une *règle statique*.  
L'article propose une *règle dynamique* qui agit à chaque étape du solveur ( itératif, coordinate descent) et pour ça, donne une suite d'ensemble qui converge (dans le sens où le diamètre tend vers 0) vers $\{ \hat{\theta (\lambda)} \}$.

## Mind the duality gap

### Les débuts

Pour une boule $B(c,r)$, on a $\mu_C(x_j)=|x_j^{T}c|+r||x_j||$.  
L'article utilise le duality gap pour trouver une suite de boules adéquates.  
En effet, pour tout couple primal/dual faisable, on a la dualité faible. Comme les conditions de Slater sont en plus vérifiées ici, on dispose en plus de la dualité forte:
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

On fixe maintenant un point $\theta \in \Delta_X$, alors $\tilde{R}_{\lambda}(\theta)= \|\theta - \frac{y}{\lambda}\| \geq \|\hat{\theta}(\lambda) - \frac{y}{\lambda}\|$ par définition de la projection.  
Ainsi la solution duale est dans couronne ou **annulus** :
$$\hat{\theta}(\lambda) \in A(\frac{y}{\lambda},\tilde{R}_{\lambda}(\theta),\hat{R}_{\lambda}(\beta))$$

### Intuition géométrique 

Le duality gap nous apprend que la solution duale $\hat{\theta}(\lambda)$ est dans une couronne centrée en $\frac{y}{\lambda}$, mais cette information reste trop grossière: une couronne n'est pas une forme géométrique simple et pratique pour notre screening.  
On va exploiter la structure convexe du problème pour trouver notre boule adéquate.

Déjà rappelons que $\hat{\theta}(\lambda)$ est la projection de $\frac{y}{\lambda}$ sur $\Delta_X$ qui est **convexe fermé**, donc $[\theta,\hat{\theta}(\lambda)]$ reste dans l'ensemble. Or comme $\hat{\theta}(\lambda)$ est le point de $\Delta_X$ le plus proche de $\frac{y}{\lambda}$, ce segment ne peut pas entrer dans la boule intérieure de la couronne, (borne donnée par le duality gap).  
$\hat{\theta}(\lambda)$ est donc relativement contraint et le point le plus éloigné de $\theta$ parmi les points compatibles avec ces contraintes est obtenu lorsque le segment partant de $\theta$ est tangent à la boule intérieure. Ce point est noté $\theta_{int}$ par l'article et donne notre rayon de sécurité.  
On obtient ainsi une boule centrée en $\theta$ et contenant la solution duale.  
Pour conclure, on obtient comme safe region 
$$
C'=B(\theta,(\tilde{R}_{\lambda}(\theta)^2 - \hat{R}_{\lambda}(\beta)^2)^{1/2})
$$
où le rayon correspond à $||\theta_{int}-\theta||$ (cf figure 1)

### qui est $\theta$ ?

Je fais un léger point sur le $\theta$ que l'on a fixé plus haut.  
Plus haut, on fait l'hypothèse que l'on dispose d'un $\theta \in \Delta_X$, il faut néanmoins en choisir un.  
Avec les conditions KKT sur le problème dual, on sait que $\hat{\theta}(\lambda)$ est proportionnel au résidu et on va donc construire à l'étape k, $\theta_k$ en fonction du résidu $\rho_k = y -X \beta_k$ et d'un constante $\alpha_k$ qu'on doit choisir de sorte que $\theta_k = \alpha_k \rho_k \in \Delta_X$.  
On choisit donc $\alpha_k$ comme la meilleure mise à l’échelle du résidu $\rho_k$ qui reste dualement faisable, autrement dit la projection du coefficient optimal non contraint sur l’intervalle de faisabilité imposé par $\Delta_X$

### Construction de la safe region

On note $r'(\theta,\beta)$ le rayon de la boule choisie.  
On a que $r'(\theta,\beta)^2 \leq r(\theta,\beta)^2:= \frac{2}{\lambda^2}G(\theta,\beta)$ où $G(\theta,\beta)$ correspond au duality gap et $r(\hat{\theta}(\lambda),\hat{\beta}(\lambda))=0$ avec la dualité forte.  

On va utiliser le duality gap comme rayon et non le rayon trouvé géométriquement, parce que celui-ci est un critère d'arrêt standard dans les solveurs et est donc déjà calculé. En effet, si $G(\theta,\beta) \leq \epsilon$, alors par dualité forte $P_{\lambda}(\beta)-P_{\lambda}(\hat{\beta}(\lambda)) \leq \epsilon$.     
C'est donc numériquement favorable.  

On note donc $C=B(\theta,r(\theta,\beta))$  
On a construit notre rayon à partir de $G(\theta,\beta)$, donc si le solveur converge vers $(\hat{\theta}(\lambda),\hat{\beta}(\lambda))$, alors le rayon tend vers 0.  
Ainsi $C_{k}=B(\theta_k,r(\theta_k,\beta_k))$ est une safe region qui converge vers $\{\hat{\theta}(\lambda)\}$ quand $\lim(\theta_k,\beta_k)=(\hat{\theta}(\lambda),\hat{\beta}(\lambda))$.  

## Implémentation

L'article donne un pseudo code qu'il faut néanmoins bien  arranger (vectorisation et pas de calculs inutiles), car avec de gros datasets (comme Leukemia qu'on va utiliser) les redondances coûtent très cher.

Vous trouverez le code exact dans mon [repo github](https://github.com/TheTrigun99/mlfromscratch/tree/main/lasso)  !

Je rappelle que nous faisons ici un algorithme pendant le solveur, mais le solveur lui ne change pas. L'algorithme de coordinate descent ne change pas, on rajoute uniquement du screening qui enlève les variables passives de telle sorte à ce que le solveur n'update que les variables actives.  

Voici le pseudo-code donné dans l'article:
![alt text](assets/img/lasso/image.png)

Je n'ai pas évoqué le warm-start plus haut, mais cela consiste à calculer la solution $\beta_{k+1}$ non pas à partir de 0, mais à partir de $\beta_k$ (possible en pratique, car pour deux valeurs proches de $\lambda$, les solutions successives du LASSO sont généralement proches. Cela rend le warm-start très efficace le long du chemin de régularisation.).

J'omet ici la fonction `__init__` qui est la même que dans mon dernier post au détail près que j'ai mis la target `y` et sa version standardisée `yc` dedans.

Le `f` du pseudo-code correspond à la fréquence à laquelle, on fait le screening. En effet, le screening a quand même un coût et il est non nécessaire de l'avoir à chaque *epoch*.

Comme suggéré par le pseudo code, on commence par calculer l'ensemble actif/passif et $\theta$:

{% highlight python %}

    def gap(self,beta,theta,a,rho):
        Gap = max((
            0.5 * np.sum((rho)**2)
            + a * np.sum(np.abs(beta))
            - 0.5 * np.sum(self.yc**2)
            + 0.5 * (a**2) * np.sum((theta - self.yc / a)**2)), 0.0)
        return Gap

    def safe_active_set(self,theta,beta,a,rho):
        gap = self.gap(beta,theta,a,rho)
        r = np.sqrt(2 * gap)/a
        scores = np.abs(self.Xc.T @ theta) + 
                r * np.sqrt(self.Xn2)

        active = np.where(scores >= 1)[0]
        z_passiv = np.where(scores < 1)[0]
        return active,z_passiv
  
    def compute_theta(self,rho,active,a):
        
        if len(active)==0:
            return np.zeros_like(rho)
        
        m = np.max( np.abs( self.Xc[:, active ].T @ rho ))
        
        if m == 0:
            return np.zeros_like(rho)
        
        alpha0 = np.dot(self.yc, rho) / ( a * np.dot(rho, rho))
        alpha = min( max(-1/m, alpha0), 1/m )
        return alpha * rho
{% endhighlight %}

`gap` calcule le duality gap et on note que l'on prend en argument le résidu $\rho$ !!!  
`safe_active_test` correspond enfin au calcul de notre safe region $C$ à l'aide des 2 fonctions ci-dessus. 

{% highlight python%}
        scores = np.abs(self.Xc.T @ theta) + 
                r * np.sqrt(self.Xn2)
        active = np.where(scores >= 1)[0]
        z_passiv = np.where(scores < 1)[0]
{% endhighlight %}
Ici, j'ai vectorisé les calculs pour la safe rule avec numpy ce qui change vraiment la donne pour le temps d'éxécution sur un gros dataset comme Leukemia où $p=7000$.

On récupère donc ensuite `active` et `z_passiv` qui nous permettent de screen les variables passives et uniquement faire la CD (coordinate descent) sur l'ensemble actif.  
On calcule ensuite $\theta$ en calculant $\alpha$ ($\rho$ est en argument).  
On va modifier le résidu en direct dans le CD et c'est pour ça qu'on le met en argument de chaque fonction ci-dessus. En effet, le calcul de $\rho$ est très coûteux quand l'on a beaucoup de variables (même si on screen entre temps), car celui-ci demande une complexité en O(n*p), tandis que l'on peut modifer le résidu incrémentalement en O(n).

{% highlight python %}
    def safe_gap_rule(self,tol,iter,bi,f,a):
        active = np.arange(self.p)
        beta = bi.copy()
        rho = self.yc - self.Xc @ beta
        theta = self.compute_theta(rho,active,a)
        for i in range(iter):
            
            if i % f == 0:

                active ,passiv = self.safe_active_set(theta,beta,a,rho)
                
                if len(passiv) > 0:
                    rho = rho + self.Xc[:, passiv] @ beta[passiv]
                    beta[passiv] = 0.0
                theta = self.compute_theta(rho,active,a)
                if len(active) == 0:
                    break
            if self.gap(beta,theta,a,rho)<tol:
                break
            for j in active:
                x_j = self.Xc[:, j]
                r_j = rho + x_j * beta[j]
                beta_new_j = self.s_threshold(np.dot(x_j, r_j), a) / self.Xn2[j]
                delta = beta_new_j - beta[j]
                rho = rho - x_j * delta
                beta[j] = beta_new_j
        return beta
{% endhighlight %}

Voici la fonction principale `safe_gap_rule` qui fait le screening et la CD.  
On commence par définir les paramètres initaux ($\rho$,$\theta$ et `active` qui correspond à l'ensemble des indices des variables actives).  
Comme dit plus haut, on ne screen pas à chaque epoch, mais uniquement lorsque `i%f==0`. 
{% highlight python%}
                if len(passiv) > 0:
                    rho = rho + self.Xc[:, passiv] @ beta[passiv]
                    beta[passiv] = 0.0
                theta = self.compute_theta(rho,active,a)
                if len(active) == 0:
                    break
{% endhighlight %} 

Ensuite, on va update le résidu incrémentalement durant l'algorithme et ne pas le recalculer entièrement ce qui nous permet d'éviter la complexité en O(n*p) !!! 

{% highlight python%}
rho = rho + self.Xc[:, passiv] @ beta[passiv]
{% endhighlight %} 

On annule les coefficients nuls dans le résidu.  
Cette étape est possible uniquement s'il y a des variables passives (`len(passiv)>0`).  
{% highlight python%}
      rho = rho - x_j * (beta_new_j-beta[j])
{%endhighlight%}
Lorsque que l'on met à jour les variables actives
De même si `len(active)==0`, on s'arrête, car on sait alors que $\beta=0$.  
Le reste correspond quasiment à la CD classique que j'ai implémenté dans mon dernier post.  
Ici, j’utilise le duality gap comme critère d’arrêt. C’est plus standard que le critère naïf basé sur la variation des coefficients, et il est déjà calculé dans l’algorithme pour construire la safe region.


### Résultats

Sur Leukemia, l’implémentation avec screening dynamique réduit fortement le temps de calcul par rapport à une coordinate descent naïve pure Python/Numpy (on passe de +10min à 80 secondes).  
Elle reste cependant nettement plus lente que sklearn, qui bénéficie d’une implémentation bas niveau très optimisée.  
Le but ici n’est donc pas de rivaliser avec sklearn en temps brut, mais de reproduire correctement l’idée de l’article et de montrer son impact concret sur le solveur.
Voici ci-dessous le lasso path de notre code qui correspond aussi à celui obtenu avec sklearn:
![image](assets/img/lasso/saferule.webp)
