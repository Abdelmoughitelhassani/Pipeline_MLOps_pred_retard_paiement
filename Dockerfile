# Image de service pour le scoring de défaut de paiement.
#
# Seules les dépendances d'inférence sont installées (requirements-serve.txt) : ni PyTorch,
# ni MLflow, ni DVC, ni CatBoost/LightGBM. Ce sont des outils d'entraînement, les embarquer
# ferait grossir l'image de plusieurs gigaoctets sans rien apporter au service.
#
# Prérequis avant build : models/final_model.joblib doit exister localement. Il est suivi
# par DVC et non par git, donc sur un dépôt fraîchement cloné il faut d'abord `dvc pull`.

FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE : pas de .pyc dans un conteneur éphémère
# PYTHONUNBUFFERED       : les logs sortent immédiatement, sans quoi ils resteraient
#                          bloqués dans le tampon et seraient invisibles dans docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Les dépendances sont installées avant la copie du code : cette couche est mise en cache
# et n'est reconstruite que si requirements-serve.txt change, pas à chaque édition du code.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Code applicatif et artefact du modèle
COPY src/ ./src/
COPY models/final_model.joblib models/model_card.json ./models/

# Exécution sans privilèges : un service exposé ne doit pas tourner en root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Sonde de vivacité. On utilise urllib plutôt que curl, absent de l'image slim :
# installer curl uniquement pour la sonde ajouterait une dépendance et une surface d'attaque.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
