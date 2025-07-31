# Application d'Imputation des Données Manquantes

## Projet Tutoré - Gestion des Valeurs Manquantes

**Étudiants :** SESSOU Jean De Dieu, AGBA DJORRE Pamela, AKAKPO Henock, ADJRA Kevin, DAOUDA SAIBOU
**Professeur :** Dr AMANA
**Groupe :** 1

---

## I. Introduction

Ce projet s'inscrit dans le cadre de la gestion et du traitement des données manquantes, un défi majeur en science des données. L'objectif était de développer une application capable de traiter automatiquement les valeurs manquantes dans n'importe quel dataset, en proposant plusieurs méthodes d'imputation et en permettant une comparaison de leurs performances.

Le projet a évolué en deux versions distinctes, reflétant notre progression dans la compréhension des besoins et l'amélioration de la solution proposée.

## II. Problématique

Les données manquantes constituent un problème récurrent dans l'analyse de données. Elles peuvent survenir pour diverses raisons : erreurs de saisie, problèmes techniques lors de la collecte, valeurs non disponibles ou données sensibles non partagées.

Ces valeurs manquantes peuvent biaiser les analyses et réduire la qualité des modèles prédictifs. Il est donc crucial de les traiter de manière appropriée avant toute analyse.

## III. Première version (V1) : L'approche initiale

La première version de notre application se concentrait sur un dataset spécifique : les données de potabilité de l'eau. Cette approche était limitée car l'application ne fonctionnait qu'avec le fichier `water_potability.csv`, les valeurs manquantes étaient identifiées manuellement, et seules quelques méthodes d'imputation étaient implémentées.

### Méthodes utilisées dans la V1

Nous avons implémenté trois approches principales :

**Traitement des outliers** : Utilisation de la méthode IQR (Interquartile Range) pour identifier et traiter les valeurs aberrantes. Cette méthode nous permettait de nettoyer les données avant l'imputation.

**Imputation simple** : Nous avons utilisé les méthodes classiques de moyenne, médiane et mode pour remplacer les valeurs manquantes. Ces méthodes sont rapides à implémenter mais peuvent introduire des biais dans les données.

**MICE Forest** : Nous avons également implémenté l'imputation par chaînes d'équations multiples avec des forêts aléatoires. Cette méthode plus sophistiquée permet de capturer les relations complexes entre les variables.

### Les limitations identifiées

Cette première approche nous a permis de comprendre les enjeux mais présentait plusieurs limitations majeures. L'application manquait de flexibilité car elle ne pouvait traiter qu'un seul dataset. L'interface utilisateur était basique, utilisant des notebooks Jupyter qui nécessitaient des connaissances techniques. Le traitement était manuel, demandant une intervention humaine à chaque étape, et la comparaison entre méthodes était limitée.

## IV. Limites de la V1 et motivation pour la V2

Après avoir testé la première version, nous avons identifié plusieurs problèmes majeurs qui nous ont poussés à repenser complètement l'architecture de l'application.

L'application était trop spécifique et ne pouvait traiter que le dataset de potabilité de l'eau. Chaque étape nécessitait une intervention manuelle, rendant le processus long et sujet aux erreurs. L'interface utilisateur était inexistante, la rendant difficile d'utilisation pour des utilisateurs non-techniques. Il était impossible de comparer efficacement les différentes méthodes d'imputation, et le traitement des outliers restait manuel.

Ces limitations nous ont fait réaliser que nous devions créer une solution plus universelle et automatisée.

## V. Deuxième version (V2) : La solution universelle

La V2 a été conçue pour répondre au besoin principal : traiter les valeurs manquantes dans n'importe quel dataset. Cette version apporte des améliorations majeures qui transforment complètement l'expérience utilisateur.

### Caractéristiques principales

L'application peut maintenant traiter n'importe quel dataset CSV, Excel ou JSON, offrant une véritable universalité. L'interface utilisateur moderne, basée sur Streamlit, est intuitive et responsive, s'adaptant à différentes tailles d'écran.

La détection automatique des valeurs manquantes et des outliers simplifie grandement le processus. Les utilisateurs peuvent également ajouter des valeurs manquantes personnalisées selon leurs besoins spécifiques.

La comparaison intégrée permet une évaluation visuelle et métrique des différentes méthodes d'imputation, facilitant le choix de la méthode la plus appropriée.

### Architecture modulaire

L'application V2 est organisée en modules spécialisés pour une meilleure maintenabilité et extensibilité :

- **DataAnalyzer** : Analyse exploratoire des données
- **MissingDetector** : Détection et configuration des valeurs manquantes
- **ImputationEngine** : Exécution des méthodes d'imputation
- **ComparisonEngine** : Comparaison des résultats
- **Visualizer** : Génération des graphiques

## VI. Méthodes d'imputation utilisées

### Imputation Simple

L'imputation simple remplace les valeurs manquantes par des statistiques descriptives. La moyenne utilise la moyenne arithmétique de la colonne, la médiane utilise la valeur centrale de la distribution, et le mode utilise la valeur la plus fréquente.

Ces méthodes sont simples à comprendre et à implémenter, mais elles peuvent biaiser les données et ignorer les relations entre variables. Elles sont particulièrement utiles pour un premier traitement rapide ou lorsque les relations entre variables ne sont pas complexes.

### Imputation KNN (K-Nearest Neighbors)

L'imputation KNN utilise les k voisins les plus proches pour estimer les valeurs manquantes. Le processus calcule la distance entre l'observation avec valeur manquante et toutes les autres, identifie les k observations les plus proches, et estime la valeur manquante basée sur ces voisins.

Cette méthode prend en compte les relations entre variables, offrant un bon compromis entre simplicité et performance. Cependant, elle est sensible au choix du paramètre k et à la qualité des données.

### MICE Forest (Multiple Imputation by Chained Equations)

MICE Forest est une méthode itérative utilisant des forêts aléatoires pour l'imputation. Le processus commence par imputer initialement les valeurs manquantes avec des méthodes simples. Pour chaque variable avec valeurs manquantes, il prédit les valeurs manquantes en utilisant les autres variables, utilisant des forêts aléatoires pour la prédiction. Le processus se répète plusieurs fois pour stabiliser les résultats.

Cette méthode est très performante et capture les relations complexes entre variables, mais elle est plus complexe et plus lente à exécuter que les autres méthodes.

## VII. Architecture de l'application

### Structure modulaire

L'application est organisée de manière modulaire pour faciliter la maintenance et l'évolution. Le fichier main.py sert de point d'entrée, utils.py contient les fonctions utilitaires, et le dossier models regroupe tous les modules spécialisés : data_analyzer.py pour l'analyse exploratoire, missing_detector.py pour la détection des valeurs manquantes, imputation_engine.py pour le moteur d'imputation, comparison_engine.py pour la comparaison des résultats, et visualizer.py pour la génération des graphiques.

### Flux de traitement

Le processus de traitement suit un flux logique et progressif :

1. **Upload des données** : Support multi-format (CSV, Excel, JSON)
2. **Analyse exploratoire** : Statistiques descriptives et visualisations
3. **Configuration** : Détection et configuration des valeurs manquantes
4. **Traitement des outliers** : Identification et gestion des valeurs aberrantes
5. **Imputation** : Application des méthodes sélectionnées
6. **Comparaison** : Évaluation et comparaison des résultats
7. **Export** : Sauvegarde des données traitées

## VIII. Fonctionnalités principales

- **Interface utilisateur intuitive** : L'interface utilisateur a été conçue pour être accessible à tous les niveaux d'expertise. Le design responsive s'adapte à différentes tailles d'écran, la navigation est
  claire avec des étapes bien définies et progressives, le feedback visuel inclut des barres de progression et des messages d'état, et l'aide contextuelle avec des tooltips
  et explications intégrées guide l'utilisateur.
- **Détection automatique** : La détection automatique simplifie grandement le processus. Les valeurs manquantes sont détectées automatiquement, incluant les valeurs NULL, NaN et autres formats
  courants. Les utilisateurs peuvent ajouter des valeurs manquantes spécifiques selon leurs besoins. Les outliers sont identifiés automatiquement avec une visualisation
  claire pour aider à la prise de décision.
- **Configuration flexible** : La configuration flexible permet une adaptation aux besoins spécifiques. La sélection de la colonne target permet d'exclure la variable à prédire du traitement
  d'imputation. Les paramètres des méthodes d'imputation sont ajustables selon les besoins. Le traitement des outliers offre plusieurs options de gestion des valeurs
  aberrantes.
- **Comparaison intégrée** : La comparaison intégrée fournit des outils d'évaluation complets. Les métriques de performance permettent une évaluation quantitative des méthodes, les visualisations
  comparatives facilitent la compréhension des différences, et l'export des résultats permet de sauvegarder les données traitées pour une utilisation ultérieure.

## IX. Résultats et comparaisons

### Métriques d'évaluation

L'application fournit plusieurs métriques pour évaluer la qualité de l'imputation :

- **Statistiques descriptives** : Moyenne, écart-type, valeurs minimales et maximales
- **Distribution des données** : Histogrammes et boxplots
- **Matrices de corrélation** : Évaluation des relations avant/après imputation
- **Stabilité** : Comparaison de la distribution des données

### Exemple de comparaison

Sur un dataset de test, nous avons observé des résultats intéressants. L'imputation simple s'est révélée rapide mais peut introduire des biais dans les données. La méthode KNN a offert un bon compromis entre vitesse et qualité d'imputation. MICE Forest a fourni la meilleure qualité d'imputation mais s'est avérée plus lente à exécuter.

### Visualisations

L'application génère automatiquement plusieurs types de visualisations pour faciliter l'analyse :

- **Boxplots** : Identification des outliers
- **Histogrammes** : Analyse des distributions
- **Matrices de corrélation** : Évaluation des relations entre variables
- **Graphiques de comparaison** : Choix entre les différentes méthodes d'imputation

## X. Conclusion

Ce projet a permis de développer une application complète et universelle pour le traitement des valeurs manquantes. Les objectifs principaux ont été atteints avec succès : l'universalité a été réalisée avec la capacité de traiter n'importe quel dataset, l'automatisation complète du processus de détection et traitement a été mise en place, l'interface utilisateur moderne et intuitive rend l'application accessible à tous, les outils de comparaison des méthodes d'imputation facilitent le choix de la méthode la plus appropriée, et la flexibilité de configuration permet une adaptation aux besoins spécifiques de chaque utilisateur.

La transition de la V1 à la V2 représente une amélioration significative dans tous les aspects du projet. Nous sommes passés d'une approche spécifique à une solution universelle, d'un traitement manuel à un processus automatisé, d'une interface technique à une interface accessible, et de fonctionnalités basiques à des outils complets de comparaison et d'export.

Pour une version future, nous pourrions envisager plusieurs améliorations comme l'intégration de méthodes avancées (deep learning), une évaluation plus robuste avec validation croisée, une interface programmatique avec API REST, le stockage des configurations dans une base de données, et des fonctionnalités de partage et collaboration.

Ce projet a permis de mettre en pratique plusieurs concepts importants de manière concrète : la gestion de projet avec planification et développement itératif, l'architecture logicielle avec design modulaire et maintenable, la conception centrée sur l'utilisateur, les méthodes d'imputation et d'évaluation en science des données, et le travail en équipe avec collaboration et répartition des tâches. L'expérience acquise sera précieuse pour de futurs projets dans ce domaine.
