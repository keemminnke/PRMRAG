#!/usr/bin/env python3
"""
Interactive JSONL Results Analyzer
Analyzes batch test results with detailed step-by-step inspection
"""

import json
import sys
from typing import List, Dict, Any
import re


class ResultsAnalyzer:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.results = []
        self.load_results()

    def load_results(self):
        """Load all results from JSONL file"""
        print(f"Loading results from {self.jsonl_path}...")
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                self.results.append(json.loads(line))
        print(f"✓ Loaded {len(self.results)} results\n")

    def show_menu(self):
        """Display interactive menu"""
        print("=" * 70)
        print("RESULTS ANALYZER MENU")
        print("=" * 70)
        print("1. Show overall statistics")
        print("2. Show incorrect answers (틀린 답변)")
        print("3. Show answers with MC stuck at 0.0")
        print("4. Show borderline cases (possibly correct but marked wrong)")
        print("5. View specific question by ID")
        print("6. Show all questions (summary)")
        print("7. Show questions by final MC range")
        print("8. Compare predicted vs gold answers")
        print("9. Export detailed report")
        print("0. Exit")
        print("=" * 70)

    def overall_stats(self):
        """Show overall statistics"""
        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)

        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        incorrect = total - correct

        with_rag = sum(1 for r in self.results if r['has_rag'])
        rag_correct = sum(1 for r in self.results if r['has_rag'] and r['is_correct'])

        cot_only = total - with_rag
        cot_correct = sum(1 for r in self.results if not r['has_rag'] and r['is_correct'])

        print(f"\nTotal questions: {total}")
        print(f"Correct: {correct} ({correct/total*100:.1f}%)")
        print(f"Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")

        print(f"\nWith RAG: {with_rag} questions")
        print(f"  Correct: {rag_correct}/{with_rag} ({rag_correct/with_rag*100:.1f}%)")

        print(f"\nCoT Only: {cot_only} questions")
        print(f"  Correct: {cot_correct}/{cot_only} ({cot_correct/cot_only*100:.1f}%)")

        # MC stuck at 0.0
        stuck_at_zero = []
        for r in self.results:
            if r['steps']:
                final_mc = r['steps'][-1]['mc_after']
                if final_mc == 0.0:
                    stuck_at_zero.append(r)

        print(f"\nMC stuck at 0.0: {len(stuck_at_zero)} questions ({len(stuck_at_zero)/total*100:.1f}%)")

    def show_incorrect(self):
        """Show all incorrect answers"""
        incorrect = [r for r in self.results if not r['is_correct']]

        print("\n" + "=" * 70)
        print(f"INCORRECT ANSWERS ({len(incorrect)} questions)")
        print("=" * 70)

        for i, r in enumerate(incorrect, 1):
            final_mc = r['steps'][-1]['mc_after'] if r['steps'] else 0.0

            print(f"\n{i}. Question ID: {r['question_id']}")
            print(f"   Question: {r['question']}")
            print(f"   Gold answer: {r['gold_answer']}")
            print(f"   Predicted: {r['predicted_answer'][:100]}...")
            print(f"   Final MC: {final_mc:.3f}")
            print(f"   Steps: {r['num_steps']} (CoT: {r['num_cot_steps']}, RAG: {r['num_rag_steps']})")

            # Show if MC was stuck
            if final_mc == 0.0:
                print(f"   ⚠️  MC stuck at 0.0!")

    def show_mc_stuck_at_zero(self):
        """Show questions where MC stayed at 0.0"""
        stuck = []
        for r in self.results:
            if r['steps']:
                final_mc = r['steps'][-1]['mc_after']
                if final_mc == 0.0:
                    stuck.append(r)

        print("\n" + "=" * 70)
        print(f"MC STUCK AT 0.0 ({len(stuck)} questions)")
        print("=" * 70)

        for i, r in enumerate(stuck, 1):
            print(f"\n{i}. Question ID: {r['question_id']}")
            print(f"   Question: {r['question']}")
            print(f"   Correct: {r['is_correct']}")
            print(f"   Predicted: {r['predicted_answer'][:100]}...")
            print(f"   Steps: {r['num_steps']} (all steps had MC=0.0)")

            # Check if all steps had MC=0.0
            all_zero = all(s['mc_after'] == 0.0 for s in r['steps'])
            if all_zero:
                print(f"   ⚠️  All {r['num_steps']} steps had MC=0.0")
            else:
                max_mc = max(s['mc_after'] for s in r['steps'])
                print(f"   Max MC during trajectory: {max_mc:.3f}")

    def show_borderline_cases(self):
        """Show cases that might be correct but marked wrong"""
        print("\n" + "=" * 70)
        print("BORDERLINE CASES (possibly correct but marked wrong)")
        print("=" * 70)

        borderline = []
        for r in self.results:
            if not r['is_correct']:
                # Check if predicted answer is similar to gold
                pred = r['predicted_answer'].lower()
                gold = str(r['gold_answer']).lower() if r['gold_answer'] else ""

                # Simple similarity check
                if gold and (gold in pred or pred in gold):
                    borderline.append(r)

        if not borderline:
            print("\n✓ No obvious borderline cases found")
            return

        print(f"\nFound {len(borderline)} potential borderline cases:")

        for i, r in enumerate(borderline, 1):
            final_mc = r['steps'][-1]['mc_after'] if r['steps'] else 0.0

            print(f"\n{i}. Question ID: {r['question_id']}")
            print(f"   Question: {r['question']}")
            print(f"   Gold answer: '{r['gold_answer']}'")
            print(f"   Predicted: '{r['predicted_answer']}'")
            print(f"   Final MC: {final_mc:.3f}")
            print(f"   💡 These might be semantically correct!")

    def view_question_details(self, question_id: str):
        """View detailed step-by-step for a specific question"""
        result = None
        for r in self.results:
            if r['question_id'] == question_id:
                result = r
                break

        if not result:
            print(f"\n❌ Question ID '{question_id}' not found")
            return

        print("\n" + "=" * 70)
        print(f"DETAILED VIEW: {result['question_id']}")
        print("=" * 70)

        print(f"\nQuestion: {result['question']}")
        print(f"Gold answer: {result['gold_answer']}")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Correct: {'✓' if result['is_correct'] else '✗'}")
        print(f"\nTotal steps: {result['num_steps']} (CoT: {result['num_cot_steps']}, RAG: {result['num_rag_steps']})")

        print("\n" + "-" * 70)
        print("STEP-BY-STEP TRAJECTORY:")
        print("-" * 70)

        for step in result['steps']:
            print(f"\nStep {step['step_num']} ({step['type'].upper()}):")
            print(f"  MC: {step['mc_before']:.3f} → {step['mc_after']:.3f} (Δ{step['mc_after']-step['mc_before']:+.3f})")
            print(f"  RPE: {step['rpe']:.3f} | Label: {step['label']}")

            if step['type'] == 'rag':
                # Extract search query
                match = re.search(r'<start_search>(.*?)<end_search>', step['content'])
                if match:
                    query = match.group(1)
                    print(f"  🔍 Query: {query}")

                # Extract intermediate answer
                if 'Intermediate answer:' in step['content']:
                    parts = step['content'].split('Intermediate answer:', 1)
                    if len(parts) > 1:
                        answer = parts[1].split('Missing info')[0].strip()
                        print(f"  📄 Answer: {answer[:150]}...")
            else:
                # CoT content
                content = step['content'].replace('\n', ' ')[:200]
                print(f"  💭 Content: {content}...")

            # Show if this step has final answer
            if 'final answer' in step['content'].lower():
                is_unknown = 'unknown' in step['content'].lower()
                if is_unknown:
                    print(f"  ⚠️  Contains 'Final Answer: unknown' (ignored)")
                else:
                    print(f"  ✓ Contains final answer")

        # Show RAG interventions
        if result['rag_interventions']:
            print("\n" + "-" * 70)
            print("RAG INTERVENTIONS:")
            print("-" * 70)
            for interv in result['rag_interventions']:
                print(f"\nStep {interv['step_num']}:")
                print(f"  MC change: {interv['mc_before']:.3f} → {interv['mc_after']:.3f}")
                print(f"  Impact: {interv.get('impact', 'N/A')}")

    def show_all_summary(self):
        """Show summary of all questions"""
        print("\n" + "=" * 70)
        print(f"ALL QUESTIONS SUMMARY ({len(self.results)} total)")
        print("=" * 70)

        for i, r in enumerate(self.results, 1):
            final_mc = r['steps'][-1]['mc_after'] if r['steps'] else 0.0
            status = "✓" if r['is_correct'] else "✗"

            question_short = r['question'][:60] + "..." if len(r['question']) > 60 else r['question']

            print(f"{i:2d}. [{status}] MC={final_mc:.3f} | {r['question_id']}")
            print(f"    Q: {question_short}")
            print(f"    A: {r['predicted_answer'][:80]}...")

    def show_by_mc_range(self):
        """Show questions grouped by final MC score"""
        print("\n" + "=" * 70)
        print("QUESTIONS BY FINAL MC RANGE")
        print("=" * 70)

        ranges = {
            'High (0.8-1.0)': [],
            'Medium (0.3-0.8)': [],
            'Low (0.0-0.3)': []
        }

        for r in self.results:
            final_mc = r['steps'][-1]['mc_after'] if r['steps'] else 0.0

            if final_mc >= 0.8:
                ranges['High (0.8-1.0)'].append(r)
            elif final_mc >= 0.3:
                ranges['Medium (0.3-0.8)'].append(r)
            else:
                ranges['Low (0.0-0.3)'].append(r)

        for range_name, questions in ranges.items():
            correct = sum(1 for q in questions if q['is_correct'])
            print(f"\n{range_name}: {len(questions)} questions")
            print(f"  Accuracy: {correct}/{len(questions)} ({correct/len(questions)*100:.1f}%)" if questions else "  No questions")

            for q in questions[:5]:  # Show first 5
                status = "✓" if q['is_correct'] else "✗"
                final_mc = q['steps'][-1]['mc_after'] if q['steps'] else 0.0
                print(f"    [{status}] MC={final_mc:.3f} | {q['question_id']}")

            if len(questions) > 5:
                print(f"    ... and {len(questions)-5} more")

    def compare_answers(self):
        """Compare predicted vs gold answers"""
        print("\n" + "=" * 70)
        print("PREDICTED vs GOLD ANSWERS COMPARISON")
        print("=" * 70)

        for i, r in enumerate(self.results, 1):
            status = "✓ CORRECT" if r['is_correct'] else "✗ INCORRECT"

            print(f"\n{i}. {status} | {r['question_id']}")
            print(f"   Question: {r['question']}")
            print(f"   Gold:      '{r['gold_answer']}'")
            print(f"   Predicted: '{r['predicted_answer']}'")

            if not r['is_correct']:
                # Try to identify why it's wrong
                pred_lower = r['predicted_answer'].lower()
                gold_lower = str(r['gold_answer']).lower() if r['gold_answer'] else ""

                if 'unknown' in pred_lower:
                    print(f"   💡 Predicted 'unknown' - couldn't find answer")
                elif gold_lower and gold_lower in pred_lower:
                    print(f"   💡 Gold answer appears in prediction - might be formatting issue")
                elif gold_lower and pred_lower in gold_lower:
                    print(f"   💡 Prediction is partial match - might be incomplete")

    def export_report(self):
        """Export detailed analysis report"""
        output_file = self.jsonl_path.replace('.jsonl', '_analysis_report.txt')

        print(f"\nExporting detailed report to: {output_file}")

        with open(output_file, 'w') as f:
            # Write overall stats
            f.write("=" * 70 + "\n")
            f.write("DETAILED ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            total = len(self.results)
            correct = sum(1 for r in self.results if r['is_correct'])

            f.write(f"Total questions: {total}\n")
            f.write(f"Correct: {correct} ({correct/total*100:.1f}%)\n")
            f.write(f"Incorrect: {total-correct} ({(total-correct)/total*100:.1f}%)\n\n")

            # Write all questions
            for i, r in enumerate(self.results, 1):
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"QUESTION {i}/{total}\n")
                f.write("=" * 70 + "\n")
                f.write(f"ID: {r['question_id']}\n")
                f.write(f"Question: {r['question']}\n")
                f.write(f"Gold: {r['gold_answer']}\n")
                f.write(f"Predicted: {r['predicted_answer']}\n")
                f.write(f"Correct: {r['is_correct']}\n")
                f.write(f"Steps: {r['num_steps']} (CoT: {r['num_cot_steps']}, RAG: {r['num_rag_steps']})\n")

                final_mc = r['steps'][-1]['mc_after'] if r['steps'] else 0.0
                f.write(f"Final MC: {final_mc:.3f}\n")

                f.write("\nStep-by-step:\n")
                for step in r['steps']:
                    f.write(f"\n  Step {step['step_num']} ({step['type']}):\n")
                    f.write(f"    MC: {step['mc_before']:.3f} → {step['mc_after']:.3f}\n")
                    f.write(f"    RPE: {step['rpe']:.3f} | Label: {step['label']}\n")

                    if step['type'] == 'rag':
                        match = re.search(r'<start_search>(.*?)<end_search>', step['content'])
                        if match:
                            f.write(f"    Query: {match.group(1)}\n")

                    content_preview = step['content'][:200].replace('\n', ' ')
                    f.write(f"    Content: {content_preview}...\n")

        print(f"✓ Report exported successfully!")

    def run(self):
        """Run interactive analyzer"""
        while True:
            self.show_menu()
            choice = input("\nEnter your choice: ").strip()

            if choice == '1':
                self.overall_stats()
            elif choice == '2':
                self.show_incorrect()
            elif choice == '3':
                self.show_mc_stuck_at_zero()
            elif choice == '4':
                self.show_borderline_cases()
            elif choice == '5':
                qid = input("Enter question ID: ").strip()
                self.view_question_details(qid)
            elif choice == '6':
                self.show_all_summary()
            elif choice == '7':
                self.show_by_mc_range()
            elif choice == '8':
                self.compare_answers()
            elif choice == '9':
                self.export_report()
            elif choice == '0':
                print("\nGoodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please try again.")

            input("\nPress Enter to continue...")


def main():
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        jsonl_path = "outputs/batch_test/results_20251124_145420.jsonl"

    analyzer = ResultsAnalyzer(jsonl_path)
    analyzer.run()


if __name__ == "__main__":
    main()
