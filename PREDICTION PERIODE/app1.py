from pydantic import BaseModel
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression  # Importez LinearRegression au lieu de Lasso
from sklearn.preprocessing import LabelEncoder  # Utilisez LabelEncoder au lieu de OneHotEncoder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Connecter à la base de données
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-RSFN8HH\SQLEXPRESS;DATABASE=immobilier;UID=sa;PWD=12356')
query = "SELECT * FROM Dimmm_immobilier"
df = pd.read_sql(query, conn)

# Convertir les colonnes en chaînes de caractères si nécessaire
df["periode_construction_prevue"] = df["periode_construction_prevue"].astype(str)
df["budget_construction_prevu"] = df["budget_construction_prevu"].astype(str)

# Créer la colonne 'caracteristiques' en concaténant les colonnes nécessaires
df["caracteristiques"] = df["periode_construction_prevue"] + " " + df["budget_construction_prevu"]

# Séparation des caractéristiques (X) et de la variable cible (y)
X = df["caracteristiques"]
y = df["periode_construction_reelle"]

# Vectorisation des caractéristiques en utilisant TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Convertir les données d'entrée en un format dense
X_vectorized_dense = X_vectorized.toarray()

# Convertir les étiquettes en valeurs numériques en utilisant LabelEncoder pour une prédiction univariée
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized_dense, y_encoded, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Définition de l'API
app = FastAPI()

# Middleware CORS pour autoriser les requêtes de n'importe quel domaine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Définition du schéma de la requête
class InputData(BaseModel):
    periode_construction_prevue: str
    budget_construction_prevu: str

# Définition de la route pour la prédiction
@app.post("/predict_p/")
async def predict_construction_period(data: InputData):
    # Récupérer les données d'entrée
    periode_construction_prevue = data.periode_construction_prevue
    budget_construction_prevu = data.budget_construction_prevu
    
    # Concaténer les caractéristiques
    caracteristiques = periode_construction_prevue + " " + budget_construction_prevu
    
    # Vectoriser les données d'entrée
    input_vectorized = vectorizer.transform([caracteristiques])
    
    # Convertir les données d'entrée en un format dense
    input_vectorized_dense = input_vectorized.toarray()
    
    # Prédire la période de construction
    predicted_period = linear_regression.predict(input_vectorized_dense)
    
    # Arrondir la valeur prédite à l'entier le plus proche
    predicted_period_value = round(predicted_period[0])
    
    # Limiter la période de construction prédite entre 1 et 20
    predicted_period_value = max(1, min(predicted_period_value, 20))
    
    return {"predicted_construction_period": predicted_period_value}