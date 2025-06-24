from pathlib import Path
import re
import json
from bs4 import BeautifulSoup
import pandas as pd
from pysentimiento.preprocessing import preprocess_tweet


def main():
    """
    Script para leer los dataset de entrenamiento, validación y test y crear un diccionario {autor: lista de tweets},
    con la ayuda de BeautifulSoup para un fácil manejo de los archivos XML en los que se encuentran los tweets de cada autor.
    Además, los tweets se preprocesan para eliminar URLs y se utiliza pysentimiento para darles formato a los tweets (e.g.,
    mapeo de emojis a palabras, etc).
    El diccionario se guarda en un archivo json para su posterior uso.
    """
    base_path = Path("./2025AuthorProfiling")

    # carpetas con tweets en xml
    train_path = base_path / "es_train"
    val_path = base_path / "es_val"
    test_path = base_path / "es_test"

    # archivos de ground truth
    train_truth_path = train_path / "truth.txt"
    val_truth_path = val_path / "truth.txt"
    test_truth_path = test_path / "truth_order.txt"

    # archivos json con todos los tweets por autor
    all_train_tweets_path = base_path / "train_tweets.json"
    all_val_tweets_path = base_path / "val_tweets.json"
    all_test_tweets_path = base_path / "test_tweets.json"

    for dir_path, authors_path, json_path in zip(
        [train_path, val_path, test_path],
        [train_truth_path, val_truth_path, test_truth_path],
        [all_train_tweets_path, all_val_tweets_path, all_test_tweets_path],
    ):
        print("Generando el archivo", json_path.name)
        # Crea un diccionario {autor: lista de tweets} a partir de los archivos XML
        tweets = create_authors_dict(dir_path, authors_path)

        # Guarda el diccionario en un archivo json
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(tweets, file, ensure_ascii=False)

        print("Archivo", json_path.name, "generado.")
        print("-" * 10)


def create_authors_dict(dir_path: Path, authors_df_path: Path) -> dict:
    """
    Crea un diccionario {autor: lista de tweets} a partir de los archivos XML en la carpeta especificada.
    """
    tweets = {}

    df = pd.read_csv(
        authors_df_path, sep=":::", names=["author", "gender", "nationality"]
    )

    for author in df["author"]:
        with open(dir_path / f"{author}.xml", "r") as file:
            soup = BeautifulSoup(file.read(), "xml")

        tweets[author] = [
            preprocess_tweet(remove_urls(doc.get_text()), user_token="usuario")
            for doc in soup.find_all("document")
        ]

    return tweets


def remove_urls(text: str) -> str:
    """
    Quita las URLs de un texto.
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    text_without_urls = url_pattern.sub("", text)

    return text_without_urls


if __name__ == "__main__":
    main()
