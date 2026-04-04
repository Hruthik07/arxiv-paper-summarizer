"""
fetch_recent_papers.py

Fetches recent arXiv papers (2024-2025) on AI Engineering topics
using the arxiv Python library. Downloads abstracts and optionally
full PDF text. Used for demo/test — showing the model summarize
a paper published last week.

Usage:
    python data/fetch_recent_papers.py --max_results 50 --output_dir ./data/recent_papers
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import arxiv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Search queries targeting AI Engineering topics
SEARCH_QUERIES = [
    "cat:cs.LG LLM fine-tuning",
    "cat:cs.AI large language model deployment",
    "cat:cs.CL instruction tuning RLHF",
    "cat:cs.LG LoRA parameter efficient fine-tuning",
    "cat:cs.AI MLOps model monitoring",
]


def fetch_papers(query: str, max_results: int = 20) -> list[dict]:
    """Fetch papers from arXiv for a given query."""
    logger.info(f"Searching arXiv: '{query}' (max={max_results})")
    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=3.0,   # be polite to the arXiv API
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        papers.append(
            {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "abstract": result.summary.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "published": result.published.isoformat(),
                "pdf_url": result.pdf_url,
                "query": query,
            }
        )
    logger.info(f"Fetched {len(papers)} papers for query: '{query}'")
    return papers


def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by arxiv_id."""
    seen = set()
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    return unique


def save_papers(papers: list[dict], output_dir: str) -> str:
    """Save papers as a JSON file. Returns the output path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"recent_papers_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(papers)} papers to {output_path}")
    return output_path


def print_sample(papers: list[dict], n: int = 3) -> None:
    """Print a few papers for quick inspection."""
    print(f"\n{'='*60}")
    print(f"Sample of {min(n, len(papers))} fetched papers:")
    print(f"{'='*60}")
    for p in papers[:n]:
        print(f"\nTitle   : {p['title']}")
        print(f"ID      : {p['arxiv_id']}")
        print(f"Published: {p['published'][:10]}")
        print(f"Abstract: {p['abstract'][:200]}...")
    print(f"{'='*60}\n")


def main(max_results_per_query: int, output_dir: str) -> None:
    all_papers = []
    for query in SEARCH_QUERIES:
        papers = fetch_papers(query, max_results=max_results_per_query)
        all_papers.extend(papers)
        time.sleep(2)  # polite delay between queries

    all_papers = deduplicate(all_papers)
    logger.info(f"Total unique papers collected: {len(all_papers)}")

    output_path = save_papers(all_papers, output_dir)
    print_sample(all_papers)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch recent arXiv AI papers")
    parser.add_argument(
        "--max_results",
        type=int,
        default=20,
        help="Max papers to fetch per query (default: 20)",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/recent_papers",
        help="Local directory to save fetched papers",
    )
    args = parser.parse_args()
    main(args.max_results, args.output_dir)
