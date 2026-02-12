import matplotlib.pyplot as plt
import numpy as np

# 데이터 설정 (5가지 케이스의 Root Cause 단계 점수)
cases = ['Case 1\n(Direction)', 'Case 2\n(Hallucination)', 'Case 3\n(Redundancy)', 'Case 4\n(Entity)', 'Case 5\n(Overconfidence)']
versaprm_scores = [0.68, 0.38, 0.82, 0.88, 0.11]  # VersaPRM 점수
critic_scores = [0.29, 0.15, 0.22, 0.15, 0.10]    # Critic (Ours) 점수 (예시: 모두 낮게 탐지함)

x = np.arange(len(cases))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# 막대 그래프 그리기
rects1 = ax.bar(x - width/2, versaprm_scores, width, label='VersaPRM (Baseline)', color='#d62728', alpha=0.8)
rects2 = ax.bar(x + width/2, critic_scores, width, label='Consensus Critic (Ours)', color='#1f77b4', alpha=0.9)

# 임계값(0.5) 라인 표시
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Acceptance Threshold (0.5)')

# 레이블 및 스타일 설정
ax.set_ylabel('Reward Score at Root Cause Step', fontsize=12)
ax.set_title('Comparison of Error Detection Capabilities across 5 Failure Modes', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(cases, fontsize=10)
ax.set_ylim(0, 1.1)
ax.legend()

# 값 표시 함수
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('all_cases_comparison.png')
print("Graph saved to all_cases_comparison.png")
