#!/usr/bin/env bash
# Deploie (ou met a jour) le service ia-free sur inference-host (inference-host, inference-host).
#   ./deploy-inference-host.sh [user@hote]     (defaut : you@inference-host)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-you@inference-host}"
IP="${HOST#*@}"
PORT="${IA_FREE_PORT:-8765}"

echo "== copie du script et de l'unite systemd =="
scp -q "$HERE/ia_free_server.py" "$HOST:/tmp/ia-free-server.new"
scp -q "$HERE/ia-free.service"    "$HOST:/tmp/ia-free.service.new"

echo "== installation sur $HOST (sudo requis) =="
ssh "$HOST" 'set -euo pipefail
  sudo install -o root -g root -m 0755 /tmp/ia-free-server.new /usr/local/bin/ia-free-server
  sudo install -o root -g root -m 0644 /tmp/ia-free.service.new /etc/systemd/system/ia-free.service
  rm -f /tmp/ia-free-server.new /tmp/ia-free.service.new
  sudo systemctl daemon-reload
  if systemctl is-active --quiet ia-free; then
    echo "ia-free deja actif -> restart"
    sudo systemctl restart ia-free
  else
    sudo systemctl enable --now ia-free
  fi
  sleep 2
  systemctl is-active --quiet ia-free && echo "service: ACTIF" || { echo "service: EN ERREUR"; exit 1; }
  journalctl -u ia-free --no-pager -n 3 | sed "s/^/  | /"'

echo "== verdict a distance =="
curl -s -m 6 "http://$IP:$PORT/free" | python3 -m json.tool
echo
echo "OK : http://$IP:$PORT/free  (GET /status, POST /lease, DELETE /lease/<id>)"
