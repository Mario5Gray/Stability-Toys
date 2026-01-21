  exec uvicorn lcm_sr_server:app \
    --host 0.0.0.0 \
    --port "${PORT:-4200}" \
    --loop uvloop \
    --http h11 \
    --timeout-keep-alive 240 \
    --log-level info
    #--no-access-log
