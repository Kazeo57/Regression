import streamlit as st
from streamlit_pdf_viewer import pdf_viewer as vw

# Titre de la page
st.header("Overview")
st.markdown("""Les données (disponibles sous Kaggle) utilisées dans le cadre de ce TP sont des données 
réelles provenant d’une compagnie d’assurance américaine X et sont utiles pour prédire les 
primes d’assurance maladie d’un citoyen américain. En effet, de nombreux facteurs 
influencent le montant des primes d’assurance maladie fixées par les compagnies d’assurance 
et qui sont indépendants de la volonté des assurés. Entre autres facteurs influençant le coût 
des primes d'assurance maladie, nous avons :
- L’âge du principal bénéficiaire (age)
- Le sexe de l'assureur (sex)
- L’indice de masse corporelle, permettant de comprendre le corps, les poids 
relativement élevés ou faibles par rapport à la taille, indice objectif de poids corporel 
(kg/m^2) utilisant le rapport taille/poids, idéalement 18,5 à 24,9 (bmi)
- Le nombre d'enfants couverts par l'assurance maladie ou le nombre de personnes à 
charge (children)
- Le statut de l’assuré par rapport au tabagisme, fumeur ou non (smoker)
- La zone résidentielle du bénéficiaire aux États-Unis, nord-est, sud-est, sud-ouest, 
nord-ouest (region)
- Les frais médicaux individuels (primes) facturés par l'assurance maladie (charges)
En tant que responsable de la cellule Informatique Décisionnelle et Gestion de Portefeuille, 
vous disposez de données mises à disposition par le département IT pour proposer un 
système intelligent de prédiction des primes d’assurance de clients désirant souscrire à un 
produit d’assurance maladie auprès de la compagnie X.""")
st.markdown("## Work")
# Chemin vers le fichier PDF (remplacez par le chemin de votre fichier PDF)
pdf_file_path = "result.pdf"

# Lecture du fichier PDF
with open(pdf_file_path, "rb") as pdf_file:
    binary_data = pdf_file.read()

# Affichage du PDF
vw(input=binary_data, width=700)

st.download_button(
    label="Télécharger le PDF",
    data=binary_data,
    file_name="document.pdf",
    mime="application/pdf"
)
