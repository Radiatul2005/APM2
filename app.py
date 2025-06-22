import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import spacy
import numpy as np
import re

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Model spaCy 'en_core_web_sm' tidak ditemukan. Silakan install dengan: python -m spacy download en_core_web_sm")
    st.stop()

# Kata-kata indikasi hoaks
trigger_words = [
    "bocor", "menggemparkan", "skandal", "konspirasi", "terbongkar",
    "geger", "heboh", "dikecam", "mencengangkan", "terungkap",
    "rahasia", "terlarang", "dilarang", "dirahasiakan", "tersembunyi"
]

def contains_trigger_word(text):
    if not text:
        return False
    return any(word.lower() in text.lower() for word in trigger_words)

def extract_person_entities(text):
    """Ekstrak nama orang dengan spaCy + regex + daftar tetap"""
    if not text:
        return set()

    found = set()

    # 1) spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            found.add(ent.text.strip().lower())

    # 2) Regex fallback: kata kapital min 3 huruf
    matches = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
    for m in matches:
        found.add(m.lower())

    # 3) Hardcoded nama-nama politikus (kalau mau)
    for keyword in ["jokowi", "prabowo", "ganjar", "anies"]:
        if keyword in text.lower():
            found.add(keyword)

    return found

def calculate_content_similarity_without_entities(text1, text2):
    """Hitung similarity konten setelah menghapus semua nama orang"""
    if not text1 or not text2:
        return 0.0
    entities1 = extract_person_entities(text1)
    entities2 = extract_person_entities(text2)
    all_entities = entities1.union(entities2)
    text1_clean = text1.lower()
    text2_clean = text2.lower()
    for entity in all_entities:
        pattern = r'\b' + re.escape(entity) + r'\b'
        text1_clean = re.sub(pattern, '', text1_clean)
        text2_clean = re.sub(pattern, '', text2_clean)
    text1_clean = re.sub(r'[^\w\s]', ' ', text1_clean)
    text2_clean = re.sub(r'[^\w\s]', ' ', text2_clean)
    text1_clean = ' '.join(text1_clean.split())
    text2_clean = ' '.join(text2_clean.split())
    if not text1_clean.strip() or not text2_clean.strip():
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([text1_clean, text2_clean])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return similarity

# ✅ FIXED auto_detect_name_substitution
def auto_detect_name_substitution(input_text, dataset_titles):
    input_entities = extract_person_entities(input_text)
    if not input_entities:
        return False, "", set(), set(), 0.0

    SIMILARITY_THRESHOLD = 0.65

    for title in dataset_titles:
        title_entities = extract_person_entities(title)
        if not title_entities:
            continue

        content_sim = calculate_content_similarity_without_entities(input_text, title)

        # Cek: mirip konten & beda nama
        if content_sim >= SIMILARITY_THRESHOLD:
            if input_entities.symmetric_difference(title_entities):
                return True, title, input_entities, title_entities, content_sim

    return False, "", set(), set(), 0.0


def extract_political_entities(text):
    """Ekstrak entitas politik dari teks menggunakan spaCy NER"""
    if not text:
        return set()
    found_entities = set()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG']:
            entity_text = ent.text.strip().lower()
            if len(entity_text) >= 2:
                found_entities.add(entity_text)
    return found_entities

def preprocess_text(text):
    if not text:
        return ""
    return text.lower().strip()

# Layout Streamlit
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")
st.title("🔍 Deteksi Berita Hoaks Politik")

st.markdown("""
Sistem ini mendeteksi hoaks berdasarkan:
- 🎯 *AUTO Name Substitution Detection*
- ✅ SVM (Support Vector Machine) dengan Cross-Validation
- ✅ Named Entity Recognition (NER)
- ✅ Cosine Similarity dengan berita valid
- ✅ Kata-kata provokatif
""")

uploaded_file = st.file_uploader("📂 Upload Dataset Excel", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        st.stop()

    if 'cleaned' in df.columns and 'label' in df.columns and 'title' in df.columns:
        st.success("✅ Dataset berhasil dimuat!")

        st.subheader("📊 Info Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Berita Valid", sum(df['label'] == 0))
        with col3:
            st.metric("Berita Hoaks", sum(df['label'] == 1))

        df = df.dropna(subset=['cleaned', 'label', 'title'])
        df = df[df['cleaned'].str.strip() != '']
        df = df[df['title'].str.strip() != '']

        df_valid = df[df['label'] == 0]
        df_hoax = df[df['label'] == 1]

        if len(df_hoax) < len(df_valid) * 0.3:
            n_samples = min(len(df_valid), len(df_hoax) * 3)
            df_hoax_upsampled = resample(df_hoax, replace=True, n_samples=n_samples, random_state=42)
            df_balanced = pd.concat([df_valid, df_hoax_upsampled])
        else:
            df_balanced = df.copy()

        X = df_balanced['cleaned']
        y = df_balanced['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.8, min_df=3, max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        base_model = LinearSVC(C=0.1, max_iter=2000, random_state=42)
        model = CalibratedClassifierCV(estimator=base_model, cv=5)
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("📊 Evaluasi Model")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Akurasi", f"{acc:.1%}")
        with col2:
            st.metric("Data Test", len(y_test))

        cm = confusion_matrix(y_test, y_pred)
        st.write("*Confusion Matrix:*")
        st.write(f"True Negative (Valid): {cm[0,0]}")
        st.write(f"False Positive (Valid→Hoaks): {cm[0,1]}")
        st.write(f"False Negative (Hoaks→Valid): {cm[1,0]}")
        st.write(f"True Positive (Hoaks): {cm[1,1]}")

        valid_titles = df_valid['title'].tolist()
        valid_title_vectors = vectorizer.transform(df_valid['title'])

        st.subheader("📝 Masukkan Berita")
        input_text = st.text_area("Tulis judul/isi berita:", height=150)

        if st.button("🔍 Deteksi Sekarang", type="primary"):
            if input_text.strip():
                name_substituted, similar_title, input_names, original_names, content_sim = auto_detect_name_substitution(input_text, valid_titles)

                if name_substituted:
                    hasil = "🚨 HOAKS CONFIRMED!"
                    alasan = f"PENGGANTIAN NAMA TERDETEKSI! Konten identik ({content_sim:.3f}) dengan berita valid tapi nama diganti"

                    st.markdown("---")
                    st.markdown(f"### 🧾 Hasil Deteksi: *{hasil}*")
                    st.error(f"{alasan}")

                    st.markdown("#### 🔍 Detail Name Substitution:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("*Input (HOAKS):*")
                        st.write(f"📰 {input_text}")
                        st.write(f"👤 Nama: {', '.join(input_names)}")
                    with col2:
                        st.write("*Original (VALID):*")
                        st.write(f"📰 {similar_title}")
                        st.write(f"👤 Nama: {', '.join(original_names)}")

                    st.metric("Similarity Konten (tanpa nama)", f"{content_sim:.3f}")

                else:
                    input_vector = vectorizer.transform([input_text])
                    similarity_scores = cosine_similarity(input_vector, valid_title_vectors).flatten()
                    max_similarity = similarity_scores.max()
                    avg_similarity = similarity_scores.mean()
                    top_indices = np.argsort(similarity_scores)[-3:][::-1]
                    top_similarities = similarity_scores[top_indices]
                    most_similar_title = valid_titles[top_indices[0]]

                    input_entities = extract_political_entities(input_text)
                    provokatif = contains_trigger_word(input_text)
                    probas = model.predict_proba(input_vector)[0]
                    prob_valid, prob_hoax = probas[0], probas[1]
                    input_cleaned = preprocess_text(input_text)
                    exact_match = any(preprocess_text(title) == input_cleaned for title in valid_titles)

                    threshold_very_similar = 0.90
                    threshold_similar = 0.80
                    threshold_moderate = 0.65

                    if exact_match:
                        if provokatif:
                            hasil = "⚠ WASPADA"
                            alasan = "Judul ditemukan di dataset valid, TAPI mengandung kata provokatif"
                        else:
                            hasil = "✅ VALID"
                            alasan = "Judul berita ditemukan persis sama di dataset valid"
                    elif max_similarity >= threshold_very_similar:
                        if provokatif:
                            hasil = "⚠ WASPADA"
                            alasan = f"Judul sangat mirip ({max_similarity:.3f}) dengan berita valid, TAPI mengandung kata provokatif"
                        else:
                            hasil = "✅ VALID"
                            alasan = f"Judul sangat mirip ({max_similarity:.3f}) dengan berita valid di dataset"
                    elif max_similarity >= threshold_similar:
                        if provokatif:
                            hasil = "⚠ WASPADA"
                            alasan = f"Judul mirip dengan berita valid ({max_similarity:.3f}) dan mengandung kata provokatif"
                        elif prob_hoax > 0.7:
                            hasil = "🔥 HOAKS"
                            alasan = f"Mirip berita valid ({max_similarity:.3f}), tapi model mendeteksi pola hoaks ({prob_hoax:.3f})"
                        else:
                            hasil = "✅ VALID"
                            alasan = f"Judul mirip dengan berita valid ({max_similarity:.3f}), tidak ada indikasi hoaks"
                    elif max_similarity >= threshold_moderate:
                        if provokatif:
                            hasil = "⚠ WASPADA"
                            alasan = f"Kemiripan sedang ({max_similarity:.3f}), mengandung kata provokatif"
                        elif prob_hoax > 0.8:
                            hasil = "🔥 HOAKS"
                            alasan = f"Model ML mendeteksi hoaks dengan confidence tinggi ({prob_hoax:.3f})"
                        elif prob_hoax > 0.6:
                            hasil = "⚠ WASPADA"
                            alasan = f"Kemiripan sedang ({max_similarity:.3f}), model mendeteksi kemungkinan hoaks ({prob_hoax:.3f})"
                        else:
                            hasil = "⚠ WASPADA"
                            alasan = f"Kemiripan cukup rendah ({max_similarity:.3f}), perlu verifikasi"
                    else:
                        if provokatif:
                            hasil = "🔥 HOAKS"
                            alasan = f"Tidak mirip dengan berita valid ({max_similarity:.3f}), mengandung kata provokatif"
                        elif prob_hoax > 0.7:
                            hasil = "🔥 HOAKS"
                            alasan = f"Tidak mirip dengan berita valid ({max_similarity:.3f}), model mendeteksi hoaks ({prob_hoax:.3f})"
                        elif prob_hoax > 0.5:
                            hasil = "⚠ WASPADA"
                            alasan = f"Tidak mirip dan ada indikasi hoaks"
                        else:
                            hasil = "⚠ WASPADA"
                            alasan = "Berita baru, perlu verifikasi manual"

                    st.markdown("---")
                    st.markdown(f"### 🧾 Hasil Deteksi: *{hasil}*")
                    st.markdown(f"*Alasan:* {alasan}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Kemiripan Maks", f"{max_similarity:.3f}")
                        st.metric("Kemiripan Rata", f"{avg_similarity:.3f}")
                    with col2:
                        st.metric("Prob. HOAKS", f"{prob_hoax:.3f}")
                        st.metric("Prob. VALID", f"{prob_valid:.3f}")

                    with st.expander("📋 Detail Analisis"):
                        st.write(f"Kata Provokatif: {'Ya' if provokatif else 'Tidak'}")
                        if provokatif:
                            found_triggers = [w for w in trigger_words if w.lower() in input_text.lower()]
                            st.write(f"Kata yang ditemukan: {', '.join(found_triggers)}")
                        st.write(f"Entitas Politik: {', '.join(input_entities) if input_entities else 'Tidak ada'}")
                        st.write("*3 Judul Paling Mirip:*")
                        for i, idx in enumerate(top_indices):
                            st.write(f"{i+1}. Similarity: {top_similarities[i]:.3f}")
                            st.write(f"   📰 {valid_titles[idx]}")
            else:
                st.warning("⚠ Masukkan teks terlebih dahulu.")
    else:
        st.error("❌ Dataset harus memiliki kolom cleaned, label, dan title.")
else:
    st.info("👆 Silakan upload dataset Excel.")