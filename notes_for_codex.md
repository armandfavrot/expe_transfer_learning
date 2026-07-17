
J'ai donc fait tourné les codes, et je trouve les résultats intéressant. Une question que je me pose est la suivante : à quelle rmse correspond l'erreur lié au bruit du modèle via le terme epsilon (avec sigma = 1) ? Et normalement, j'imagine que s'il y a suffisament d'observation pour l'entrainement, les mlp devraient atteindre cette erreur ? 

Ok, alors peux tu mettre à jour le protocole avec n = 2000 au lieu de 1000 pour la taille des jeux de données simulées, et ensuite doubler la taille de tous les jeux de données qui en découle, sauf la taille des jeux de données des échantillons cibles : pour le moment on teste 10, 50, 100, il faudrait changer par 10, 50, 100, 200.











