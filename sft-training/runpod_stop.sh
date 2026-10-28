#!/usr/bin/env bash

trainer_pid="$(pgrep -o -f '[p]ython.*trainer\.py[[:space:]]+config\.yaml')"

if [ -z "$trainer_pid" ]; then
    echo "Trainer process not found."
    exit 1
fi

if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "RunPod credentials missing."
    exit 1
fi

nohup bash -c '
pid="$1"
report="/workspace/v5/sft-training/outputs/v5-sft-01/training_report.json"

echo "Watching trainer PID ${pid}"

while kill -0 "$pid" 2>/dev/null; do
    sleep 30
done

if grep -Eq "\"status\"[[:space:]]*:[[:space:]]*\"complete\"" "$report" 2>/dev/null; then
    echo "Training completed successfully."
else
    echo "Training failed or was interrupted."
fi

echo "Syncing files and stopping Pod."
sync

for attempt in 1 2 3 4 5; do
    curl --fail --show-error --silent \
      --connect-timeout 10 \
      --max-time 30 \
      --request POST \
      --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" &&
      exit 0

    echo "Stop attempt ${attempt} failed; retrying in 15 seconds."
    sleep 15
done

echo "Unable to stop the Pod after five attempts."
' _ "$trainer_pid" > auto-stop.log 2>&1 &

echo "Watcher PID: $!"