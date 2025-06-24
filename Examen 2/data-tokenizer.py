from pathlib import Path
import json
from nltk.tokenize import TweetTokenizer


def main():
    """
    Script para leer los dataset de entrenamiento, validación y test en formato json (creados por el script data-preprocess.py)
    y crear otros archivos json con los tweets de cada autor tokenizados por TweetTokenizer.
    """
    base_path = Path("./2025AuthorProfiling")

    # archivos json con todos los tweets por autor
    all_train_tweets_path = base_path / "train_tweets.json"
    all_val_tweets_path = base_path / "val_tweets.json"
    all_test_tweets_path = base_path / "test_tweets.json"

    train_tk_path = base_path / "train_tk.json"
    val_tk_path = base_path / "val_tk.json"
    test_tk_path = base_path / "test_tk.json"

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    for tweets_json_path, tweets_tk_json_path in zip(
        [all_train_tweets_path, all_val_tweets_path, all_test_tweets_path],
        [train_tk_path, val_tk_path, test_tk_path],
    ):
        print("Generando el archivo", tweets_tk_json_path.name)
        with open(tweets_json_path, "r", encoding="utf-8") as file:
            tweets = json.load(file)

        # Tokeniza los tweets de cada autor
        corpus_tk = {
            author: [tokenizer.tokenize(tweet) for tweet in tweets[author]]
            for author in tweets
        }

        # Guarda el diccionario en un archivo json
        with open(tweets_tk_json_path, "w", encoding="utf-8") as file:
            json.dump(corpus_tk, file, ensure_ascii=False)
        print("Archivo", tweets_tk_json_path.name, "generado.")
        print("-" * 10)


if __name__ == "__main__":
    main()
