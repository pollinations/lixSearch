#!/usr/bin/env bash
# lixSearch Landing Page — Build & Deploy
#
# Usage:
#   ./deploy.sh              — build + deploy
#   ./deploy.sh build        — build only (static export to ./out/)
#   ./deploy.sh deploy       — deploy only (requires prior build)
#   ./deploy.sh install-node — install Node 22 LTS if not present

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="lixsearch"
REQUIRED_NODE_MAJOR=22

# ── Helpers ──────────────────────────────────────────

log()  { echo -e "\033[0;34mℹ\033[0m $*"; }
ok()   { echo -e "\033[0;32m✓\033[0m $*"; }
fail() { echo -e "\033[0;31m✗\033[0m $*" >&2; exit 1; }

check_node() {
  local node_ver
  node_ver=$(node --version 2>/dev/null | sed 's/^v//' | cut -d. -f1)

  if [ -z "$node_ver" ]; then
    fail "Node.js not found. Run: ./deploy.sh install-node"
  fi

  if [ "$node_ver" -gt "$REQUIRED_NODE_MAJOR" ]; then
    echo ""
    echo "  ⚠  Node v$(node --version) detected — Next.js requires Node ≤22 LTS."
    echo "     Node 25's experimental localStorage API breaks SSR builds."
    echo ""
    echo "  Fix: Run './deploy.sh install-node' to install Node 22 LTS,"
    echo "  then use 'nvm use 22' or update your PATH."
    echo ""
    fail "Incompatible Node version"
  fi
}

check_deps() {
  if [ ! -d "node_modules" ]; then
    log "Installing dependencies..."
    npm install --prefer-offline --no-audit
  fi
}

# ── Commands ──────────────────────────────────────────

do_install_node() {
  log "Installing Node.js 22 LTS..."

  if command -v nvm &>/dev/null || [ -s "$HOME/.nvm/nvm.sh" ]; then
    # nvm is available
    [ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
    nvm install 22
    nvm use 22
    ok "Node $(node --version) installed via nvm"
    echo "  Run 'nvm use 22' before building, or add it to your shell profile."
  else
    # Install nvm first
    log "Installing nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
    nvm install 22
    nvm use 22
    ok "Node $(node --version) installed via nvm"
    echo ""
    echo "  Add to your shell profile (~/.bashrc or ~/.zshrc):"
    echo '    export NVM_DIR="$HOME/.nvm"'
    echo '    [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"'
    echo ""
  fi
}

do_build() {
  check_node
  check_deps

  log "Building static site..."
  npx next build

  if [ ! -d "out" ]; then
    fail "Build failed — ./out/ not created"
  fi

  ok "Build complete — output in ./out/"
  echo "  $(find out -type f | wc -l) files, $(du -sh out | cut -f1) total"
}

do_deploy() {
  if [ ! -d "out" ]; then
    fail "./out/ not found. Run './deploy.sh build' first."
  fi

  log "Deploying to Cloudflare Pages (project: $PROJECT)..."
  npx wrangler pages deploy ./out --project-name "$PROJECT"
  ok "Deploy complete"
}

# ── Main ──────────────────────────────────────────────

if [ $# -eq 0 ]; then
  do_build
  do_deploy
  exit 0
fi

for cmd in "$@"; do
  case "$cmd" in
    build)        do_build ;;
    deploy)       do_deploy ;;
    install-node) do_install_node ;;
    *)
      echo "Usage: ./deploy.sh [build|deploy|install-node]"
      echo ""
      echo "  build        — build static site to ./out/"
      echo "  deploy       — deploy ./out/ to Cloudflare Pages"
      echo "  install-node — install Node 22 LTS via nvm"
      echo ""
      echo "  No args = build + deploy"
      exit 1
      ;;
  esac
done
