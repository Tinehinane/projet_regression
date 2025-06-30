# Projet de Régression Linéaire par Descente de Gradient

Ce projet implémente une régression linéaire univariée en utilisant la méthode de descente de gradient. Le modèle est entraîné sur des données générées aléatoirement et évalué à l'aide du coefficient de détermination (R²).

## Fonctionnalités
- Génération de données synthétiques avec `make_regression`
- Initialisation aléatoire des paramètres du modèle
- Calcul du modèle linéaire : $y = X \cdot \theta$
- Implémentation de la fonction de coût (MSE)
- Descente de gradient avec historique des coûts
- Visualisation des résultats et de la courbe d'apprentissage
- Calcul du coefficient de détermination R²

## Résultats Clés
- Paramètres finaux optimaux : $\theta = [193.30, 2.03]$
- Coefficient de détermination : $R^2 = 0.989$ (performance excellente)
- Visualisations :
  - Droite de régression avant/après entraînement
  - Courbe de convergence de la fonction de coût

## Comment Utiliser
1. Installer les dépendances : `pip install -r requirements.txt`
2. Exécuter le notebook Jupyter `regression_lineaire.ipynb`
3. Les sections sont organisées dans l'ordre suivant :
   - Génération et visualisation des données
   - Initialisation du modèle
   - Définition des fonctions (modèle, coût, gradient)
   - Entraînement par descente de gradient
   - Évaluation et visualisation des résultats

## Fonctions Principales
```python
def model(X, theta):
    return X.dot(theta)

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def grad_descent(X, y, theta, learning_rate, n_iterations):
    # Implémentation de l'algorithme
