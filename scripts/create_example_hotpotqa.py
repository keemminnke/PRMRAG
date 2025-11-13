#!/usr/bin/env python3
"""
Create example HotpotQA data for testing adaptive trajectory generation.
"""

import jsonlines
from pathlib import Path


def create_example_hotpotqa_data():
    """Create minimal HotpotQA-style data for testing."""

    # Example questions
    questions = [
        {
            "_id": "hotpot_001",
            "question": "What is the capital of France and what is its population?",
            "answer": "Paris, approximately 2.2 million",
            "supporting_facts": [
                ["Paris", 0],
                ["Paris", 1],
            ],
            "context": [
                ["Paris", [
                    "Paris is the capital and most populous city of France.",
                    "Paris has a population of approximately 2.2 million people.",
                ]],
                ["France", [
                    "France is a country in Western Europe.",
                    "The capital of France is Paris.",
                ]],
            ],
        },
        {
            "_id": "hotpot_002",
            "question": "Who wrote Harry Potter and when was the first book published?",
            "answer": "J.K. Rowling, 1997",
            "supporting_facts": [
                ["J.K. Rowling", 0],
                ["Harry Potter and the Philosopher's Stone", 0],
            ],
            "context": [
                ["J.K. Rowling", [
                    "J.K. Rowling is a British author, best known for the Harry Potter series.",
                    "She was born in 1965.",
                ]],
                ["Harry Potter and the Philosopher's Stone", [
                    "Harry Potter and the Philosopher's Stone was first published in 1997.",
                    "It is the first novel in the Harry Potter series.",
                ]],
            ],
        },
        {
            "_id": "hotpot_003",
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter",
            "supporting_facts": [
                ["Jupiter", 0],
            ],
            "context": [
                ["Jupiter", [
                    "Jupiter is the largest planet in our solar system.",
                    "It is a gas giant with a mass greater than all other planets combined.",
                ]],
                ["Solar System", [
                    "The solar system consists of the Sun and objects that orbit it.",
                    "There are eight planets in the solar system.",
                ]],
            ],
        },
    ]

    # Create corpus (all documents from contexts)
    corpus = []
    doc_id = 0

    for q in questions:
        for title, sentences in q['context']:
            corpus.append({
                'id': f"doc_{doc_id}",
                'title': title,
                'text': ' '.join(sentences),
            })
            doc_id += 1

    return questions, corpus


def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    questions, corpus = create_example_hotpotqa_data()

    # Save questions
    questions_path = output_dir / "example_hotpotqa_questions.jsonl"
    print(f"Creating {len(questions)} example questions...")
    with jsonlines.open(questions_path, mode='w') as writer:
        for q in questions:
            writer.write(q)
    print(f"✓ Saved to {questions_path}")

    # Save corpus
    corpus_path = output_dir / "example_hotpotqa_corpus.jsonl"
    print(f"\nCreating corpus with {len(corpus)} documents...")
    with jsonlines.open(corpus_path, mode='w') as writer:
        for doc in corpus:
            writer.write(doc)
    print(f"✓ Saved to {corpus_path}")

    print(f"\nExample usage:")
    print(f"  python scripts/generate_adaptive_trajectories.py \\")
    print(f"    --config configs/adaptive_generation.yaml \\")
    print(f"    --questions {questions_path} \\")
    print(f"    --corpus {corpus_path} \\")
    print(f"    --output data/generated/example_adaptive.jsonl \\")
    print(f"    --num-trajectories 2")


if __name__ == "__main__":
    main()
