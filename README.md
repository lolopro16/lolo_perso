# lolo_perso

## Dashboard spécifique au Nasdaq

Ce projet propose un dashboard interactif construit avec [Streamlit](https://streamlit.io/) pour suivre l'indice **Nasdaq Composite** et ses principaux acteurs.

### Fonctionnalités
- Visualisation des cours (bougies ou courbes) du Nasdaq Composite.
- Indicateurs clés : variation quotidienne, performance depuis le début de l'année, volatilité annualisée.
- Comparaison de performance avec un panier de grandes capitalisations du Nasdaq.
- Analyse sectorielle des principales positions du tracker Nasdaq-100 (QQQ).

### Installation
1. Créez un environnement virtuel (optionnel mais recommandé).
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

### Lancement
Exécutez la commande suivante puis ouvrez l'URL affichée dans le terminal :
```bash
streamlit run nasdaq_dashboard/app.py
```

### Remarques
- Les données de marché proviennent de l'API Yahoo Finance via la librairie `yfinance`.
- Une connexion internet est nécessaire pour actualiser les données.
