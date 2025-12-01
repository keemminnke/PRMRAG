#!/usr/bin/env python3
"""
Compare evaluation methods: Paper's F1 vs Our Token Overlap
"""
import string
import re
from collections import Counter


# ===== Paper's Implementation =====
def paper_normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def paper_qa_f1_score(prediction, ground_truth):
    prediction_tokens = paper_normalize_answer(prediction).split()
    ground_truth_tokens = paper_normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paper_exact_match(prediction, ground_truth):
    return paper_normalize_answer(prediction) == paper_normalize_answer(ground_truth)


def paper_match(prediction, ground_truths):
    """Check if any ground truth is substring of prediction"""
    for gt in ground_truths:
        if gt in prediction:
            return True
    return False


# ===== Our Implementation =====
def our_normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\b(sir|mr|mrs|ms|miss|dr|prof|professor)\b\.?', ' ', text)
    text = re.sub(r"'s\b", '', text)
    text = re.sub(r"s'\b", 's', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    normalized_words = []
    for word in words:
        if len(word) > 3 and word.endswith('s') and word not in ['yes', 'no', 'class', 'glass', 'mass', 'pass', 'boss', 'ross', 'moss', 'loss', 'toss']:
            singular = word[:-1]
            normalized_words.append(singular)
        else:
            normalized_words.append(word)

    text = ' '.join(normalized_words)
    text = ' '.join(text.split())
    return text.strip()


def our_check_answer_match(predicted: str, gold: str, threshold: float = 0.66) -> bool:
    predicted = our_normalize_answer(predicted)
    gold = our_normalize_answer(gold)

    if predicted == gold:
        return True

    if gold.strip() in ['yes', 'no']:
        if gold.strip() == 'no':
            negative_indicators = ['not', 'no', 'false', 'incorrect', 'never', 'neither', 'none']
            pred_words = predicted.split()
            if any(indicator in pred_words for indicator in negative_indicators):
                return True
        elif gold.strip() == 'yes':
            positive_indicators = ['yes', 'correct', 'true', 'indeed', 'certainly']
            pred_words = predicted.split()
            if any(indicator in pred_words for indicator in positive_indicators):
                return True

    if gold in predicted:
        return True

    gold_tokens = gold.split()
    pred_tokens = predicted.split()

    if not gold_tokens:
        return False

    matches = sum(1 for token in gold_tokens if token in pred_tokens)
    accuracy = matches / len(gold_tokens)

    if accuracy >= 0.66:
        return True

    return accuracy >= threshold


# ===== Test Cases =====
test_cases = [
    # (prediction, gold_answer, description)
    ("Arthur's Magazine", "Arthur's Magazine", "Exact match with possessive"),
    ("Arthur's Magazine was started first", "Arthur's Magazine", "Extra words"),
    ("Victor John Mature", "Victor Mature", "Middle name"),
    ("Sir Francis Nethersole", "Francis Nethersole", "Title"),
    ("The directors voted", "director", "Plural vs singular"),
    ("President Richard Nixon's middle name", "Milhouse", "Wrong answer"),
    ("Yes, that is correct", "yes", "Yes/No question"),
    ("No, that's not right", "no", "Yes/No question (no)"),
    ("6.213 km long", "6.213 km", "Extra word"),
    ("Pavel Urysohn was correct", "yes", "Yes answer in sentence"),
]


print("=" * 80)
print("EVALUATION METHOD COMPARISON")
print("=" * 80)
print()

paper_correct = 0
our_correct = 0
disagreements = []

for i, (pred, gold, desc) in enumerate(test_cases, 1):
    # Paper's method (using F1 > 0.5 as threshold)
    f1_score = paper_qa_f1_score(pred, gold)
    exact_match = paper_exact_match(pred, gold)
    paper_result = exact_match or f1_score > 0.5

    # Our method
    our_result = our_check_answer_match(pred, gold)

    # Count
    if paper_result:
        paper_correct += 1
    if our_result:
        our_correct += 1

    # Check disagreement
    if paper_result != our_result:
        disagreements.append((i, pred, gold, desc, paper_result, our_result, f1_score))

    print(f"[Test {i}] {desc}")
    print(f"  Prediction: '{pred}'")
    print(f"  Gold: '{gold}'")
    print(f"  Paper (F1={f1_score:.3f}): {'✅' if paper_result else '❌'}")
    print(f"  Our (66% overlap): {'✅' if our_result else '❌'}")
    if paper_result != our_result:
        print(f"  ⚠️  DISAGREEMENT")
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Paper's method: {paper_correct}/{len(test_cases)} correct")
print(f"Our method: {our_correct}/{len(test_cases)} correct")
print(f"Disagreements: {len(disagreements)}")
print()

if disagreements:
    print("=" * 80)
    print("DETAILED DISAGREEMENTS")
    print("=" * 80)
    for test_num, pred, gold, desc, paper_res, our_res, f1 in disagreements:
        print(f"\n[Test {test_num}] {desc}")
        print(f"  Prediction: '{pred}'")
        print(f"  Gold: '{gold}'")
        print(f"  Paper F1: {f1:.3f} → {'CORRECT' if paper_res else 'WRONG'}")
        print(f"  Our method → {'CORRECT' if our_res else 'WRONG'}")
        print(f"  Who's right: ", end="")

        # Manual judgment
        if test_num in [1, 2, 3, 4, 5, 7, 9, 10]:  # These should be correct
            expected = True
        elif test_num == 6:  # Wrong answer
            expected = False
        else:
            expected = None

        if expected is not None:
            if our_res == expected and paper_res != expected:
                print("✅ OUR METHOD")
            elif paper_res == expected and our_res != expected:
                print("✅ PAPER METHOD")
            else:
                print("⚖️  Both have issues")
