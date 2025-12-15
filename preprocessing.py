"""
Modul preprocessing teks untuk deteksi kemiripan judul tugas akhir.
Menggunakan Sastrawi untuk stemming Bahasa Indonesia.
"""

import re
from typing import List
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopwords Bahasa Indonesia (bisa diperluas)
STOPWORDS_ID = set("""
yang dan di ke dari pada untuk dengan sebagai atau oleh sebuah adalah 
seperti dalam antara karena guna tersebut terhadap sangat lebih memiliki 
dapat telah masih juga kami kita adanya sehingga bukan namun agar serta 
ini itu saja akan ada bila mana oleh lebih kurang hanya bagi sudah
""".split())


def preprocess_text(text: str) -> str:
    """
    Preprocessing teks judul skripsi:
    1. Ubah ke huruf kecil
    2. Hapus tanda baca dan angka (hanya huruf dan spasi)
    3. Hapus stopwords
    4. Stemming menggunakan Sastrawi
    
    Args:
        text: Teks judul mentah
        
    Returns:
        Teks yang sudah dipreprocess (string token yang sudah di-stem)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Hapus tanda baca dan angka, hanya huruf dan spasi
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Split menjadi token dan hapus stopwords serta token pendek (<=1 karakter)
    tokens = [t for t in text.split() if t and t not in STOPWORDS_ID and len(t) > 1]
    
    # Stemming setiap token
    stems = [stemmer.stem(t) for t in tokens]
    
    # Gabung kembali menjadi string
    return " ".join(stems)


def preprocess_corpus(titles: List[str]) -> List[str]:
    """
    Preprocessing batch untuk korpus judul.
    
    Args:
        titles: List judul mentah
        
    Returns:
        List judul yang sudah dipreprocess
    """
    return [preprocess_text(title) for title in titles]
