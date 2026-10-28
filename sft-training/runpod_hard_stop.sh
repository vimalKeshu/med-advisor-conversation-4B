nohup bash -c '
sleep 6h
sync
curl --fail --silent --show-error \
  --request POST \
  --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
' > /workspace/v5/sft-training/hard-stop.log 2>&1 &

echo "Hard-stop fallback PID: $!"