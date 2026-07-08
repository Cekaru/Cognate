# syntax=docker/dockerfile:1

# --- build stage ---------------------------------------------------------
FROM golang:1.25-alpine AS build
WORKDIR /src

# Copy the module graph first so dependency download caches in its own layer.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/proxy ./cmd/proxy

# --- runtime stage (distroless, nonroot) ---------------------------------
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/proxy /proxy
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/proxy"]
