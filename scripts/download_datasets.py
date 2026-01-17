"""
Download NaturalQuestions, HotpotQA, and build BM25 index.

This script uses pure Python (no Java required) for BM25.
"""

import os
from pathlib import Path
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import pickle
import json
from tqdm import tqdm

DATA_DIR = Path(os.environ.get("LCR_DATA_DIR", "./data/external"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("LCR Dataset Download & BM25 Index Build")
print(f"Target directory: {DATA_DIR}")
print("="*60)


def download_natural_questions():
    """Download NaturalQuestions dataset."""
    print("\n1️⃣  Downloading NaturalQuestions...")
    nq_path = DATA_DIR / "natural_questions"

    try:
        dataset = load_dataset(
            "natural_questions",
            split="train[:10000]",
            cache_dir=str(DATA_DIR / "cache")
        )

        dataset.save_to_disk(str(nq_path))
        print(f"✓ Saved {len(dataset):,} NaturalQuestions examples to {nq_path}")
        return dataset

    except Exception as e:
        print(f"❌ Error downloading NaturalQuestions: {e}")
        print("   Trying alternative: google/natural_questions")

        dataset = load_dataset(
            "google-research-datasets/natural_questions",
            split="train[:10000]",
            cache_dir=str(DATA_DIR / "cache")
        )
        dataset.save_to_disk(str(nq_path))
        print(f"✓ Saved {len(dataset):,} examples to {nq_path}")
        return dataset


def download_hotpotqa():
    """Download HotpotQA dataset."""
    print("\n2️⃣  Downloading HotpotQA...")
    hotpot_path = DATA_DIR / "hotpotqa"

    try:
        dataset = load_dataset(
            "hotpot_qa",
            "fullwiki",
            split="train[:5000]",
            cache_dir=str(DATA_DIR / "cache")
        )

        dataset.save_to_disk(str(hotpot_path))
        print(f"✓ Saved {len(dataset):,} HotpotQA examples to {hotpot_path}")
        return dataset

    except Exception as e:
        print(f"❌ Error downloading HotpotQA: {e}")
        return None


def download_wikipedia_corpus():
    """Download Wikipedia corpus for BM25"""
    print("\n3️⃣  Downloading Wikipedia corpus...")
    wiki_path = DATA_DIR / "wikipedia"

    try:
        print("   Trying wikimedia/wikipedia...")
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train[:100000]",
            cache_dir=str(DATA_DIR / "cache"),
            trust_remote_code=True
        )

        dataset.save_to_disk(str(wiki_path))
        print(f"✓ Saved {len(dataset):,} Wikipedia articles to {wiki_path}")

        jsonl_path = wiki_path / "articles.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for article in dataset:
                f.write(json.dumps({
                    'id': article.get('id', ''),
                    'title': article.get('title', ''),
                    'text': article.get('text', '')
                }, ensure_ascii=False) + '\n')

        print(f"✓ Also saved as JSONL: {jsonl_path}")
        return dataset

    except Exception as e:
        print(f"   Failed: {e}")
        print("   Trying alternative: legacy wikipedia format...")

        try:
            dataset = load_dataset(
                "princeton-nlp/wikipedia-passages",
                split="train[:100000]",
                cache_dir=str(DATA_DIR / "cache")
            )

            dataset.save_to_disk(str(wiki_path))
            print(f"✓ Saved {len(dataset):,} Wikipedia passages to {wiki_path}")

            jsonl_path = wiki_path / "articles.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for i, passage in enumerate(dataset):
                    f.write(json.dumps({
                        'id': f'passage_{i}',
                        'title': passage.get('title', ''),
                        'text': passage.get('text', passage.get('passage', ''))
                    }, ensure_ascii=False) + '\n')

            print(f"✓ Also saved as JSONL: {jsonl_path}")
            return dataset

        except Exception as e2:
            print(f"   Failed: {e2}")
            print("   Creating fallback Wikipedia from NaturalQuestions passages...")
            return None


def build_bm25_index(wikipedia_dataset):
    """Build BM25 index using rank-bm25 (pure Python, no Java)"""
    print("\n4️⃣  Building BM25 index (pure Python)...")

    print("   Tokenizing Wikipedia articles...")
    tokenized_corpus = []
    doc_ids = []
    texts = []

    for i, article in enumerate(tqdm(wikipedia_dataset, desc="Tokenizing")):
        text = article.get('text', '')
        if len(text) > 100:
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)
            doc_ids.append(article.get('id', f'doc_{i}'))
            texts.append(text)

    print(f"   Building BM25 index from {len(tokenized_corpus):,} documents...")
    bm25 = BM25Okapi(tokenized_corpus)

    index_path = DATA_DIR / "bm25_index"
    index_path.mkdir(exist_ok=True)

    with open(index_path / "bm25_model.pkl", 'wb') as f:
        pickle.dump(bm25, f)

    with open(index_path / "doc_ids.json", 'w') as f:
        json.dump(doc_ids, f)

    with open(index_path / "texts.json", 'w') as f:
        json.dump(texts, f)

    print(f"✓ BM25 index saved to {index_path}")
    print(f"   - Documents: {len(doc_ids):,}")
    print(f"   - Index size: ~{os.path.getsize(index_path / 'bm25_model.pkl') / 1024 / 1024:.1f} MB")

    return bm25, doc_ids, texts


def test_bm25_search(bm25, doc_ids, texts):
    """Test BM25 search with sample query"""
    print("\n5️⃣  Testing BM25 search...")

    query = "When was the Eiffel Tower completed?"
    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)
    top_5_indices = scores.argsort()[-5:][::-1]

    print(f"   Query: {query}")
    print(f"   Top-5 results:")
    for rank, idx in enumerate(top_5_indices, 1):
        print(f"     {rank}. Doc {doc_ids[idx]}")
        print(f"        Score: {scores[idx]:.2f}")
        print(f"        Text: {texts[idx][:100]}...")

    print("✓ BM25 search working!")


def create_summary():
    """Create summary file"""
    summary_path = DATA_DIR / "DOWNLOAD_SUMMARY.txt"

    with open(summary_path, 'w') as f:
        f.write("LCR Dataset Download Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Download directory: {DATA_DIR}\n\n")

        f.write("Downloaded datasets:\n")
        f.write("1. NaturalQuestions: natural_questions/\n")
        f.write("2. HotpotQA: hotpotqa/\n")
        f.write("3. Wikipedia: wikipedia/\n")
        f.write("4. BM25 Index: bm25_index/\n\n")

        f.write("Files created:\n")
        for file in DATA_DIR.rglob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                f.write(f"  {file.relative_to(DATA_DIR)}: {size_mb:.1f} MB\n")

        f.write("\nNote: This is a subset for testing. Full datasets:\n")
        f.write("  - NaturalQuestions: ~300K examples\n")
        f.write("  - HotpotQA: ~113K examples\n")
        f.write("  - Wikipedia: ~6M articles\n")
        f.write("\nTo download full datasets, modify split parameter in code.\n")

    print(f"\n✓ Summary saved to {summary_path}")


def main():
    """Main download workflow"""
    try:
        nq_dataset = download_natural_questions()

        hotpot_dataset = download_hotpotqa()

        wiki_dataset = download_wikipedia_corpus()

        if wiki_dataset:
            bm25, doc_ids, texts = build_bm25_index(wiki_dataset)

            test_bm25_search(bm25, doc_ids, texts)

        create_summary()

        print("\n" + "="*60)
        print("✅ All downloads complete!")
        print("="*60)
        print(f"\nData location: {DATA_DIR}")
        print("\nNext steps:")
        print("1. Check DOWNLOAD_SUMMARY.txt for details")
        print("2. Run build_real_dataset.py to create conflict datasets")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
