# VeriNewsBot

Bienvenido/a a este repositorio 👋.  
Aquí encontrarás todo lo necesario para entrenar, evaluar y reproducir modelos de **detección de noticias falsas** utilizando el conjunto de datos _Fake News Detection Dataset_.  

> **¿Por qué importa?**  
> En la era digital, la desinformación se propaga más rápido que nunca. Este proyecto ofrece un punto de partida robusto para crear sistemas que ayuden a distinguir entre noticias reales y fabricadas, aportando así a la integridad de la información.

---

## 📂 Estructura del repositorio

```text
.
├── data/
│   ├── Fake.csv
│   └── True.csv
├── notebooks/
│   ├── ModeloOscar.ipynb       # Primer pipeline de modelado (baseline clásico)
│   └── ModeloWEmbeddings.ipynb # Pipeline con embeddings y modelos avanzados
├── embeddings/
│   ├── ada_embeddings.json       # Embeddings
│   └── ada_embeddings_2.json     # Embeddings
├── src/
│   └── utils.py          # Funciones auxiliares de carga y preprocesamiento
├── outputs/
│   ├── models/           # Modelos entrenados y serializados
│   └── reports/          # Métricas y gráficas de resultados
└── README.md
```

---

## 📊 Dataset

| Atributo | Descripción                                                   | Tipo |
|----------|---------------------------------------------------------------|------|
| `title`  | Titular del artículo                                          | Texto |
| `text`   | Cuerpo completo de la noticia                                 | Texto |
| `subject`| Categoría o tema (p. ej. política, mundo, deporte…)          | Categórico |
| `date`   | Fecha de publicación                                          | Fecha |

- **True.csv** – 21 417 artículos **reales**.  
- **Fake.csv** – 23 481 artículos **falsos**.

### Posibles casos de uso
- Entrenamiento de modelos NLP binarios (**fake vs real**).
- Análisis de sentimiento y temático de la desinformación.
- Exploración de patrones lingüísticos entre noticias auténticas y engañosas.

---

## 🛠️ Requisitos

- Python ≥ 3.9  
- Jupyter Notebook o JupyterLab
