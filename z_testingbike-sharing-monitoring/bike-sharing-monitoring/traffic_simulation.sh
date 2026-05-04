#!/usr/bin/env bash
# traffic_simulation.sh
#
# Generates /predict traffic against the bike-api so that the API
# Performance dashboard panels (request rate, latency, error rate)
# show real data.
#
# Usage:
#   bash traffic_simulation.sh           # default: 60s, 0.3s spacing
#   bash traffic_simulation.sh 120 0.1   # duration_seconds  inter_request_delay

set -u

URL="${URL:-http://localhost:8080/predict}"
DURATION="${1:-60}"
DELAY="${2:-0.3}"

# Hand-picked feature combinations covering different seasons/hours/weather.
PAYLOADS=(
'{"season":1,"holiday":0,"workingday":1,"weathersit":1,"temp":0.24,"atemp":0.2879,"hum":0.81,"windspeed":0.0,"mnth":1,"hr":0,"weekday":6}'
'{"season":1,"holiday":0,"workingday":1,"weathersit":2,"temp":0.30,"atemp":0.31,"hum":0.65,"windspeed":0.10,"mnth":2,"hr":8,"weekday":1}'
'{"season":2,"holiday":0,"workingday":1,"weathersit":1,"temp":0.50,"atemp":0.48,"hum":0.55,"windspeed":0.15,"mnth":4,"hr":17,"weekday":3}'
'{"season":3,"holiday":0,"workingday":0,"weathersit":1,"temp":0.72,"atemp":0.70,"hum":0.45,"windspeed":0.08,"mnth":7,"hr":12,"weekday":0}'
'{"season":4,"holiday":1,"workingday":0,"weathersit":3,"temp":0.40,"atemp":0.38,"hum":0.85,"windspeed":0.30,"mnth":11,"hr":19,"weekday":5}'
)

echo "Generating /predict traffic against $URL for ${DURATION}s..."
END=$((SECONDS + DURATION))
COUNT=0
ERRORS=0
while [ $SECONDS -lt $END ]; do
    P="${PAYLOADS[$((COUNT % ${#PAYLOADS[@]}))]}"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL" \
        -H 'Content-Type: application/json' \
        -d "$P" || echo "000")
    if [ "$HTTP_CODE" != "200" ]; then
        ERRORS=$((ERRORS + 1))
    fi
    COUNT=$((COUNT + 1))
    sleep "$DELAY"
done

echo "Done. Sent $COUNT requests (errors: $ERRORS)."
