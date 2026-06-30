# FTSE100 Data

Los archivos de datos del FTSE100 no están incluidos en el repositorio porque superan el límite de 100 MB de GitHub:

- `dataframes_Train_scrapeadas.csv` — 557 MB
- `dataframes_Test_scrapeadas.csv` — 140 MB

Para reproducir los experimentos de cross-training, coloca estos archivos en esta carpeta (`ftse100/data/`) antes de ejecutar los scripts.

**Estos archivos ya contienen todas las features NLP pre-computadas** (EmoLex, Empath, NLTK SIA, similitudes ODI/JDI basadas en RoBERTa, densidades POS). No es necesario recomputarlas — los scripts de cross-training los usan directamente.

Los archivos se obtuvieron mediante web scraping de Glassdoor para empresas del FTSE-100 (Bolsa de Londres), con las mismas features NLP aplicadas que a los datos SP500. Si necesitas recomputar las features desde datos crudos, consulta los scripts en `ftse100/`.
