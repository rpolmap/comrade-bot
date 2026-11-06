FROM rust:1-slim AS build
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:stable-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/comrade-bot /usr/local/bin/comrade-bot
# Mount a persistent volume at /data so /setup keys survive restarts
ENV KEYS_FILE=/data/keys.json
CMD ["comrade-bot"]
