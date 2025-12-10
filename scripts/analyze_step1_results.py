#!/usr/bin/env python3
"""Step 1 결과 분석 스크립트.

사용법:
    python scripts/analyze_step1_results.py --input outputs/hybrid_1000q_from_0/results_hybrid_XXXXXX.jsonl

옵션:
    --input: 결과 JSONL 파일 경로
    --show-all: 모든 케이스 출력 (기본: 요약만)
    --show-correct: 정답 케이스만 출력
    --show-incorrect: 오답 케이스만 출력
    --show-fp: False Positive 의심 케이스
    --show-fn: False Negative 의심 케이스
    --show-rag-fail: RAG 실패 케이스
    --show-format-issues: 양식 미준수 케이스
    --limit N: 출력 개수 제한 (기본: 전체)
    --export-csv: CSV로 내보내기
"""

import sys
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.utils.answer_utils import check_answer_match, normalize_answer


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def color(text, c):
    return f"{c}{text}{Colors.ENDC}"


def load_results(input_file):
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def print_divider(char='=', length=80):
    print(char * length)


def print_question_detail(r, idx, show_steps=True):
    """한 질문의 상세 정보 출력"""
    print_divider()
    print(f"{color(f'[{idx}]', Colors.BOLD)} {color('정답' if r['is_correct'] else '오답', Colors.GREEN if r['is_correct'] else Colors.RED)}")
    print_divider('-')
    print(f"{color('Q:', Colors.CYAN)} {r['question']}")
    print(f"{color('Gold:', Colors.GREEN)} {r['gold_answer']}")
    print(f"{color('Pred:', Colors.YELLOW)} {r['predicted_answer']}")

    # Normalized 비교
    gold_norm = normalize_answer(r['gold_answer'])
    pred_norm = normalize_answer(r['predicted_answer'])
    print(f"{color('Gold(norm):', Colors.GREEN)} {gold_norm}")
    print(f"{color('Pred(norm):', Colors.YELLOW)} {pred_norm}")

    # 토큰 매칭
    gold_tokens = set(gold_norm.split())
    pred_tokens = set(pred_norm.split())
    matched = gold_tokens & pred_tokens
    missing = gold_tokens - pred_tokens
    print(f"{color('Token Match:', Colors.CYAN)} {matched} | Missing: {missing}")

    # RAG 사용 여부
    rag_steps = [s for s in r['steps'] if s.get('action') == 'Search']
    print(f"{color('Steps:', Colors.CYAN)} {len(r['steps'])}개 (RAG: {len(rag_steps)}개)")

    if show_steps:
        print_divider('-')
        print(color('Step Details:', Colors.BOLD))
        for i, step in enumerate(r['steps']):
            action = step.get('action', 'Reason')
            label = step.get('label', 'N/A').upper()
            rpe = step.get('rpe', 0)
            mc_before = step.get('mc_before', 0)
            mc_after = step.get('mc_after', 0)

            label_color = Colors.GREEN if label == 'GOOD' else Colors.RED
            action_color = Colors.CYAN if action == 'Search' else Colors.YELLOW

            print(f"\n  {color(f'Step {i+1}:', Colors.BOLD)} [{color(action, action_color)}] {color(label, label_color)}")
            print(f"    RPE: {rpe:.3f} | MC: {mc_before:.3f} → {mc_after:.3f} ({mc_after-mc_before:+.3f})")

            content = step.get('content', '')[:300]
            # 줄바꿈 처리
            content_lines = content.split('\n')[:5]
            for line in content_lines:
                if line.strip():
                    print(f"    {line[:100]}")

    print()


def analyze_summary(results):
    """전체 요약 통계"""
    print_divider('=')
    print(color('STEP 1 RESULTS SUMMARY', Colors.BOLD + Colors.BLUE))
    print_divider('=')

    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])

    # RAG vs CoT
    rag_questions = [r for r in results if any(s.get('action') == 'Search' for s in r['steps'])]
    cot_questions = [r for r in results if not any(s.get('action') == 'Search' for s in r['steps'])]

    rag_correct = sum(1 for r in rag_questions if r['is_correct'])
    cot_correct = sum(1 for r in cot_questions if r['is_correct'])

    print(f"\n{color('1. Overall Accuracy', Colors.BOLD)}")
    print(f"   Total: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"   RAG Used: {rag_correct}/{len(rag_questions)} ({rag_correct/len(rag_questions)*100:.1f}%)" if rag_questions else "   RAG Used: N/A")
    print(f"   CoT Only: {cot_correct}/{len(cot_questions)} ({cot_correct/len(cot_questions)*100:.1f}%)" if cot_questions else "   CoT Only: N/A")

    # MC 기반 난이도별
    mc_easy = [r for r in results if r['steps'] and r['steps'][0].get('mc_after', 0) >= 0.7]
    mc_mid = [r for r in results if r['steps'] and 0.3 <= r['steps'][0].get('mc_after', 0) < 0.7]
    mc_hard = [r for r in results if r['steps'] and r['steps'][0].get('mc_after', 0) < 0.3]

    print(f"\n{color('2. Accuracy by Difficulty (Initial MC)', Colors.BOLD)}")
    print(f"   Easy (MC≥0.7): {sum(1 for r in mc_easy if r['is_correct'])}/{len(mc_easy)} ({sum(1 for r in mc_easy if r['is_correct'])/max(len(mc_easy),1)*100:.1f}%)")
    print(f"   Medium (0.3≤MC<0.7): {sum(1 for r in mc_mid if r['is_correct'])}/{len(mc_mid)} ({sum(1 for r in mc_mid if r['is_correct'])/max(len(mc_mid),1)*100:.1f}%)")
    print(f"   Hard (MC<0.3): {sum(1 for r in mc_hard if r['is_correct'])}/{len(mc_hard)} ({sum(1 for r in mc_hard if r['is_correct'])/max(len(mc_hard),1)*100:.1f}%)")

    # Step-level 라벨
    all_steps = []
    for r in results:
        for step in r['steps']:
            all_steps.append({
                'label': step.get('label', '').lower(),
                'rpe': step.get('rpe', 0),
                'action': step.get('action', 'Reason'),
                'q_correct': r['is_correct'],
            })

    good_steps = [s for s in all_steps if s['label'] == 'good']
    bad_steps = [s for s in all_steps if s['label'] == 'bad']

    print(f"\n{color('3. Step-level Labels', Colors.BOLD)}")
    print(f"   Total Steps: {len(all_steps)}")
    print(f"   GOOD: {len(good_steps)} ({len(good_steps)/len(all_steps)*100:.1f}%)")
    print(f"   BAD: {len(bad_steps)} ({len(bad_steps)/len(all_steps)*100:.1f}%)")

    # GOOD 라벨 품질
    good_from_correct = sum(1 for s in good_steps if s['q_correct'])
    bad_from_incorrect = sum(1 for s in bad_steps if not s['q_correct'])

    print(f"\n{color('4. Label Quality', Colors.BOLD)}")
    print(f"   GOOD from correct questions: {good_from_correct}/{len(good_steps)} ({good_from_correct/max(len(good_steps),1)*100:.1f}%)")
    print(f"   BAD from incorrect questions: {bad_from_incorrect}/{len(bad_steps)} ({bad_from_incorrect/max(len(bad_steps),1)*100:.1f}%)")

    # RAG 효과
    rag_steps = [s for s in all_steps if s['action'] == 'Search']
    print(f"\n{color('5. RAG Steps', Colors.BOLD)}")
    print(f"   Total RAG Steps: {len(rag_steps)}")
    if rag_steps:
        rag_good = sum(1 for s in rag_steps if s['label'] == 'good')
        print(f"   GOOD: {rag_good}/{len(rag_steps)} ({rag_good/len(rag_steps)*100:.1f}%)")

    print()


def find_false_positives(results):
    """틀렸는데 맞았다고 판정된 케이스 찾기"""
    fps = []
    for r in results:
        if r['is_correct']:
            # 재검증
            new_check = check_answer_match(r['predicted_answer'], r['gold_answer'])
            if not new_check:
                fps.append(r)
    return fps


def find_false_negatives(results):
    """맞았는데 틀렸다고 판정된 케이스 찾기"""
    fns = []
    for r in results:
        if not r['is_correct']:
            # 재검증
            new_check = check_answer_match(r['predicted_answer'], r['gold_answer'])
            if new_check:
                fns.append(r)
    return fns


def find_edge_cases(results):
    """경계 케이스 (부분 매칭) 찾기"""
    edge_cases = []
    for r in results:
        if not r['is_correct']:
            gold_norm = normalize_answer(r['gold_answer'])
            pred_norm = normalize_answer(r['predicted_answer'])
            gold_tokens = set(gold_norm.split())
            pred_tokens = set(pred_norm.split())
            overlap = gold_tokens & pred_tokens
            if overlap and len(overlap) < len(gold_tokens):
                edge_cases.append({
                    **r,
                    'overlap': overlap,
                    'missing': gold_tokens - pred_tokens,
                })
    return edge_cases


def find_format_issues(results):
    """양식 미준수 케이스 찾기"""
    issues = []
    for r in results:
        for step in r['steps']:
            if step.get('action') == 'Search':
                content = step.get('content', '')
                step_issues = []

                if not re.search(r'\[\d+\]', content):
                    step_issues.append('no_citation')
                if not re.search(r'Observation:', content, re.IGNORECASE):
                    step_issues.append('no_observation')
                if not re.search(r'Sub-answer:', content, re.IGNORECASE):
                    step_issues.append('no_subanswer')

                if step_issues:
                    issues.append({
                        'question': r['question'],
                        'gold': r['gold_answer'],
                        'issues': step_issues,
                        'content': content[:500],
                    })
    return issues


def find_rag_failures(results):
    """RAG 실패 케이스 분석"""
    failures = {
        'mc_zero': [],
        'mc_degraded': [],
        'retrieval_miss': [],
    }

    for r in results:
        if not r['is_correct']:
            has_rag = any(s.get('action') == 'Search' for s in r['steps'])
            if has_rag:
                # MC=0 케이스
                mc_values = [s.get('mc_after', 0) for s in r['steps']]
                if all(m == 0 for m in mc_values):
                    failures['mc_zero'].append(r)
                    continue

                # MC 악화 케이스
                for step in r['steps']:
                    if step.get('action') == 'Search':
                        mc_change = step.get('mc_after', 0) - step.get('mc_before', 0)
                        if mc_change < -0.1:
                            failures['mc_degraded'].append(r)
                            break
                else:
                    # 검색 실패
                    failures['retrieval_miss'].append(r)

    return failures


def export_csv(results, output_path):
    """CSV로 내보내기"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'question', 'gold_answer', 'predicted_answer',
            'is_correct', 'num_steps', 'num_rag_steps',
            'initial_mc', 'final_mc', 'has_citation'
        ])

        for i, r in enumerate(results):
            rag_steps = [s for s in r['steps'] if s.get('action') == 'Search']
            initial_mc = r['steps'][0].get('mc_after', 0) if r['steps'] else 0
            final_mc = r['steps'][-1].get('mc_after', 0) if r['steps'] else 0

            # Citation 확인
            has_citation = any(
                re.search(r'\[\d+\]', s.get('content', ''))
                for s in rag_steps
            )

            writer.writerow([
                i, r['question'], r['gold_answer'], r['predicted_answer'],
                r['is_correct'], len(r['steps']), len(rag_steps),
                initial_mc, final_mc, has_citation
            ])

    print(f"CSV exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 1 결과 분석")
    parser.add_argument("--input", required=True, help="결과 JSONL 파일 경로")
    parser.add_argument("--show-all", action="store_true", help="모든 케이스 출력")
    parser.add_argument("--show-correct", action="store_true", help="정답 케이스만")
    parser.add_argument("--show-incorrect", action="store_true", help="오답 케이스만")
    parser.add_argument("--show-fp", action="store_true", help="False Positive 의심")
    parser.add_argument("--show-fn", action="store_true", help="False Negative 의심")
    parser.add_argument("--show-edge", action="store_true", help="경계 케이스 (부분매칭)")
    parser.add_argument("--show-rag-fail", action="store_true", help="RAG 실패 케이스")
    parser.add_argument("--show-format-issues", action="store_true", help="양식 미준수")
    parser.add_argument("--limit", type=int, default=None, help="출력 개수 제한")
    parser.add_argument("--no-steps", action="store_true", help="스텝 상세 숨기기")
    parser.add_argument("--export-csv", type=str, help="CSV 내보내기 경로")
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} results\n")

    # Export CSV if requested
    if args.export_csv:
        export_csv(results, args.export_csv)
        return

    # 기본: 요약 출력
    analyze_summary(results)

    # 특정 필터 옵션
    show_steps = not args.no_steps
    limit = args.limit

    if args.show_all:
        print_divider('=')
        print(color('ALL CASES', Colors.BOLD))
        for i, r in enumerate(results[:limit] if limit else results):
            print_question_detail(r, i+1, show_steps)

    if args.show_correct:
        correct_cases = [r for r in results if r['is_correct']]
        print_divider('=')
        print(color(f'CORRECT CASES ({len(correct_cases)})', Colors.BOLD + Colors.GREEN))
        for i, r in enumerate(correct_cases[:limit] if limit else correct_cases):
            print_question_detail(r, i+1, show_steps)

    if args.show_incorrect:
        incorrect_cases = [r for r in results if not r['is_correct']]
        print_divider('=')
        print(color(f'INCORRECT CASES ({len(incorrect_cases)})', Colors.BOLD + Colors.RED))
        for i, r in enumerate(incorrect_cases[:limit] if limit else incorrect_cases):
            print_question_detail(r, i+1, show_steps)

    if args.show_fp:
        fps = find_false_positives(results)
        print_divider('=')
        print(color(f'FALSE POSITIVE SUSPECTS ({len(fps)})', Colors.BOLD + Colors.RED))
        for i, r in enumerate(fps[:limit] if limit else fps):
            print_question_detail(r, i+1, show_steps)

    if args.show_fn:
        fns = find_false_negatives(results)
        print_divider('=')
        print(color(f'FALSE NEGATIVE SUSPECTS ({len(fns)})', Colors.BOLD + Colors.YELLOW))
        for i, r in enumerate(fns[:limit] if limit else fns):
            print_question_detail(r, i+1, show_steps)

    if args.show_edge:
        edges = find_edge_cases(results)
        print_divider('=')
        print(color(f'EDGE CASES - Partial Match ({len(edges)})', Colors.BOLD + Colors.YELLOW))
        for i, r in enumerate(edges[:limit] if limit else edges):
            print_question_detail(r, i+1, show_steps)
            print(f"   {color('Matched tokens:', Colors.GREEN)} {r['overlap']}")
            print(f"   {color('Missing tokens:', Colors.RED)} {r['missing']}")

    if args.show_rag_fail:
        failures = find_rag_failures(results)
        print_divider('=')
        print(color('RAG FAILURE ANALYSIS', Colors.BOLD + Colors.RED))

        print(f"\n{color('MC=0 from start:', Colors.CYAN)} {len(failures['mc_zero'])} cases")
        for i, r in enumerate(failures['mc_zero'][:5]):
            print(f"  [{i+1}] {r['question'][:60]}... → {r['gold_answer']}")

        print(f"\n{color('MC Degraded by RAG:', Colors.CYAN)} {len(failures['mc_degraded'])} cases")
        for i, r in enumerate(failures['mc_degraded'][:5]):
            print(f"  [{i+1}] {r['question'][:60]}... → {r['gold_answer']}")

        print(f"\n{color('Other RAG failures:', Colors.CYAN)} {len(failures['retrieval_miss'])} cases")

    if args.show_format_issues:
        issues = find_format_issues(results)
        print_divider('=')
        print(color(f'FORMAT ISSUES ({len(issues)})', Colors.BOLD + Colors.YELLOW))
        for i, issue in enumerate(issues[:limit] if limit else issues):
            print(f"\n[{i+1}] {issue['question'][:60]}...")
            print(f"   Issues: {issue['issues']}")
            print(f"   Content: {issue['content'][:200]}...")


if __name__ == "__main__":
    main()
