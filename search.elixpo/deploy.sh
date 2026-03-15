#!/usr/bin/env bash
# Usage:
#   ./deploy.sh              — build + deploy
#   ./deploy.sh build        — build only
#   ./deploy.sh deploy       — deploy only (must build first)

set -euo pipefail

PROJECT="lixsearch"

do_build() {
  echo "=== Building static site ==="
  npm run build
  echo "Build complete — output in ./out/"
  echo ""
}

do_deploy() {
  if [ ! -d "out" ]; then
    echo "Error: ./out/ not found. Run './deploy.sh build' first."
    exit 1
  fi

  echo "=== Deploying to Cloudflare Pages ==="
  npx wrangler pages deploy ./out --project-name "$PROJECT"
  echo "Deploy complete."
  echo ""
}

# No args = build + deploy
if [ $# -eq 0 ]; then
  do_build
  do_deploy
  exit 0
fi

for cmd in "$@"; do
  case "$cmd" in
    build)  do_build ;;
    deploy) do_deploy ;;
    *)
      echo "Unknown command: $cmd"
      echo "Usage: ./deploy.sh [build] [deploy]"
      exit 1
      ;;
  esac
done
