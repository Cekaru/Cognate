# syntax=docker/dockerfile:1

# --- build stage ---------------------------------------------------------
FROM golang:1.23-alpine AS build
WORKDIR /src

# Phase 0 uses only the Go standard library, so go.mod alone is enough and
# there is no go.sum yet. Copy the module graph first for layer caching.
COPY go.mod ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/proxy ./cmd/proxy

# --- runtime stage (distroless, nonroot) ---------------------------------
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/proxy /proxy
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/proxy"]
