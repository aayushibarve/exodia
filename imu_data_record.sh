#!/bin/bash

DURATION=$((4*60*60))  # 4 hours

echo "Starting rosbag recording..."

ros2 bag record /imu/data_raw -o rosbag_test_4h_final &
BAG_PID=$!

START=$(date +%s)

# Give rosbag time to print its startup messages
sleep 5

while kill -0 "$BAG_PID" 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [ "$ELAPSED" -ge "$DURATION" ]; then
        printf "\nStopping recording...\n"
        kill -INT "$BAG_PID"
        wait "$BAG_PID"
        break
    fi

    REMAINING=$((DURATION - ELAPSED))

    printf "\rElapsed: %02d:%02d:%02d | Remaining: %02d:%02d:%02d" \
        $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) \
        $((REMAINING/3600)) $(((REMAINING%3600)/60)) $((REMAINING%60))

    sleep 1
done

printf "\nDone.\n"
