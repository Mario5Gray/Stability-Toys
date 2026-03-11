docker network create appnet

docker run -d \
  --name redis \
  --hostname redis \
  --network appnet \
  -p 6379:6379 \
  redis:7
