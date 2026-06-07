#!/bin/bash
# Build and run wally dev container with local repo mounted using Podman
podman-compose up --build -d
podman exec -it wally-dev bash
