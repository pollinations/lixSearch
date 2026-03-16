#!/usr/bin/env bash
# lixSearch Landing Page — Build static site for nginx
#
# Usage:
#   ./deploy.sh              — install deps + build
#   ./deploy.sh install-node — install Node 22 LTS via nvm

set -euo pipefail
cd "$(dirname "$0")"

REQUIRED_NODE_MAJOR=22

log()  { echo -e "\033[0;34mℹ\033[0m $*"; }
ok()   { echo -e "\033[0;32m✓\033[0m $*"; }
fail() { echo -e "\033[0;31m✗\033[0m $*" >&2; exit 1; }

check_node() {
  local node_ver
  node_ver=$(node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1)
  [ -z "$node_ver" ] && fail "Node.js not found. Run: ./deploy.sh install-node"
  [ "$node_ver" -gt "$REQUIRED_NODE_MAJOR" ] && fail "Node $(node --version) detected — Next.js needs ≤22 LTS. Run: ./deploy.sh install-node"
}

do_install_node() {
  log "Installing Node.js 22 LTS via nvm..."
  if [ -s "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
  else
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    source "$NVM_DIR/nvm.sh"
  fi
  nvm install 22
  nvm use 22
  ok "Node $(node --version) ready"
}

do_build() {
  check_node

  if [ ! -d "node_modules" ]; then
    log "Installing dependencies..."
    npm install --prefer-offline --no-audit
  fi

  log "Building static site..."
  npx next build

  [ ! -d "out" ] && fail "Build failed — ./out/ not created"
  ok "Build complete — $(find out -type f | wc -l) files, $(du -sh out | cut -f1)"
  echo "  Restart nginx/docker to pick up changes."
}

case "${1:-build}" in
  build)        do_build ;;
  install-node) do_install_node ;;
  *)            echo "Usage: ./deploy.sh [build|install-node]"; exit 1 ;;
esac
