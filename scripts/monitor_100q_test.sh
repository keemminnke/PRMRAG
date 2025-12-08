#!/bin/bash

LOG_FILE="/tmp/hybrid_100q_test.log"
OUTPUT_DIR="outputs/hybrid_100q_from_11"

echo "=========================================="
echo "100 Questions Test Monitor"
echo "Started at: $(date)"
echo "=========================================="

while true; do
    echo ""
    echo "=== Check at $(date +%H:%M:%S) ==="

    # Check if process is running
    if ps aux | grep -q "[p]ython3 scripts/batch_test_hybrid.py --num-questions 100 --start-idx 10"; then
        echo "✓ Process is running"

        # Check GPU usage
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "GPU Status:"
            nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
                awk -F', ' '{printf "  Memory: %s/%s MB (%.1f%%), GPU: %s%%\n", $1, $2, ($1/$2*100), $3}'
        fi

        # Check log file size
        if [ -f "$LOG_FILE" ]; then
            LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
            LOG_LINES=$(wc -l < "$LOG_FILE")
            echo "  Log: $LOG_SIZE ($LOG_LINES lines)"

            # Show last few lines
            echo ""
            echo "Last 5 log lines:"
            tail -5 "$LOG_FILE" | sed 's/^/    /'
        else
            echo "  Log file not yet created"
        fi

        # Check output directory
        if [ -d "$OUTPUT_DIR" ]; then
            RESULT_FILE=$(find "$OUTPUT_DIR" -name "results_*.jsonl" -type f 2>/dev/null | head -1)
            if [ -n "$RESULT_FILE" ]; then
                QUESTION_COUNT=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo "0")
                echo "  Progress: $QUESTION_COUNT/100 questions completed"
            fi
        fi

    else
        echo "✗ Process finished or stopped"

        # Check if results exist
        if [ -d "$OUTPUT_DIR" ]; then
            RESULT_FILE=$(find "$OUTPUT_DIR" -name "results_*.jsonl" -type f 2>/dev/null | head -1)
            if [ -n "$RESULT_FILE" ]; then
                QUESTION_COUNT=$(wc -l < "$RESULT_FILE")
                echo "  Final count: $QUESTION_COUNT questions"
            fi
        fi

        echo ""
        echo "Test completed at: $(date)"
        break
    fi

    # Wait before next check
    sleep 60
done
