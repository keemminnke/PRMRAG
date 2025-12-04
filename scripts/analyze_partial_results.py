#!/usr/bin/env python3
"""중간 결과 실시간 분석 스크립트"""

import json
import sys
from pathlib import Path

def analyze_partial_results(results_file: str):
    """부분 결과 분석"""

    if not Path(results_file).exists():
        print(f"⚠️  결과 파일이 아직 없습니다: {results_file}")
        return

    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except:
                    pass

    if not results:
        print("⚠️  아직 결과가 없습니다.")
        return

    n = len(results)
    correct = sum(1 for r in results if r.get('is_correct', False))

    print("="*80)
    print(f"📊 중간 결과 분석 (처리된 질문: {n}개)")
    print("="*80)

    # 정확도
    acc = correct / n * 100 if n > 0 else 0
    print(f"\n✅ 정확도: {correct}/{n} ({acc:.1f}%)")

    # Max steps 분석
    max_steps_count = sum(1 for r in results if r['num_steps'] >= 10)
    max_steps_pct = max_steps_count / n * 100 if n > 0 else 0
    print(f"⚠️  Max steps 도달: {max_steps_count}/{n} ({max_steps_pct:.1f}%)")

    # 평균 스텝 수
    avg_steps = sum(r['num_steps'] for r in results) / n if n > 0 else 0
    print(f"📈 평균 스텝 수: {avg_steps:.1f}")

    # 최근 10개 질문
    print(f"\n{'='*80}")
    print(f"최근 10개 질문 결과:")
    print("="*80)

    for i, r in enumerate(results[-10:], start=max(1, n-9)):
        status = "✅" if r.get('is_correct', False) else "❌"
        print(f"[{i}/{n}] {status} {r['num_steps']} steps - {r['question'][:60]}...")
        print(f"      Gold: {r['gold_answer'][:50]}")
        print(f"      Pred: {r['predicted_answer'][:50]}")
        print()

    # RAG 통계
    total_rag_steps = sum(len([s for s in r.get('steps', []) if s.get('type') == 'rag']) for r in results)
    print(f"{'='*80}")
    print(f"RAG 통계:")
    print("="*80)
    print(f"총 RAG steps: {total_rag_steps}")
    print(f"질문당 평균 RAG steps: {total_rag_steps / n:.1f}")

    # 틀린 질문 패턴 분석
    incorrect = [r for r in results if not r.get('is_correct', False)]
    if incorrect:
        print(f"\n{'='*80}")
        print(f"❌ 틀린 질문 패턴 ({len(incorrect)}개):")
        print("="*80)

        # 스텝 수별 분포
        step_buckets = {
            '1-3': 0,
            '4-6': 0,
            '7-9': 0,
            '10': 0
        }

        for r in incorrect:
            steps = r['num_steps']
            if steps <= 3:
                step_buckets['1-3'] += 1
            elif steps <= 6:
                step_buckets['4-6'] += 1
            elif steps <= 9:
                step_buckets['7-9'] += 1
            else:
                step_buckets['10'] += 1

        print("스텝 수별 분포:")
        for bucket, count in step_buckets.items():
            pct = count / len(incorrect) * 100 if incorrect else 0
            print(f"  {bucket} steps: {count}개 ({pct:.1f}%)")

if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/batch_test_100q_final/results_*.jsonl"

    # Glob pattern 처리
    if '*' in results_file:
        from glob import glob
        files = glob(results_file)
        if files:
            results_file = sorted(files)[-1]  # 가장 최근 파일

    analyze_partial_results(results_file)
