import streamlit as st
import fitz  # PyMuPDF

# Titre de la page
st.title("Affichage d'un document PDF")

# Chemin vers le fichier PDF
pdf_file_path = "chemin/vers/votre/document.pdf"

# Ouverture du fichier PDF
pdf_document = fitz.open(pdf_file_path)

# Affichage de chaque page du PDF
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    st.image(img_data)

# Fermeture du document PDF
pdf_document.close()
