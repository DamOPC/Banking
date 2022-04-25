# Projet7

Implémentez un modèle de scoring!

[logo_entreprise](https://user-images.githubusercontent.com/90404008/161977409-df361004-a96f-4a64-acf9-92e724101f35.png)

L'objectif est de développer un modèle de scoring de la probabilité de défaut de paiement d'un client pour étayer la décision d'accorder ou non un prêt à un client potentiel

Les données sont plusieurs fichiers '.csv' qui renseignent des informations concernant les clients d'une banque.

Tous les fichiers .csv sont téléchargeables à l'adresse suivante : https://www.kaggle.com/c/home-credit-default-risk/data

Le dossier ne contient pas les fichiers compte tenu du volume de ceux-ci. Mais elles ressemblent à ceci :

![home_credit](https://user-images.githubusercontent.com/90404008/161977543-ec67af64-1e64-4958-af1a-5a70b0d9117d.png)

Le dossier contient 3 fichiers .ipynb. Le premier correspond à la partie pré-traitement des données et est nommé Projet_6_preprocessing.ipynb. Dans cette partie, nous avons traité les valeurs abérrantes, manquantes, fait du feature engineering et encodés les variables catégorielles et cycliques.

Le deuxième fichier concerne la partie modélisation, nommé Projet_6_modelling.ipynb. Il s'agit de la séléction du modèle, de l'optimisation des hyper-paramètres ainsi que de l'analyse des résultats.

Le troisième fichier, quant à lui, concerne le déploiement du dashboard avec dash, une sur-couche de flask.

Il y a également le fichier correspondant au modèle qui est facilement importable depuis n'importe où.

Pour finir, il a une note méthodologique concernant la partie modélisation pour aider les chargés de client à comprendre le travail effectué.
