#!/bin/bash
# Periodically prunes unused Docker networks (docker network prune -f only
# touches networks with zero attached containers, so this is safe to run
# alongside active generate_loop.py runs).
#
# Exists because docker-compose-created networks (two per pier eval task:
# _default and _pier-egress-internal) don't get torn down by `docker rm -f`
# on their containers -- only `docker compose down` does that. Every
# interrupted `pier run` (a killed generate_loop.py, a killed meta-agent
# hang) leaves its task containers' networks orphaned. Confirmed live: this
# silently zeroed out two real generations' worth of eval scores
# (deep_swe_final gen_5 and gen_6 both scored 0.0 -- every task failed with
# "all predefined address pools have been fully subnetted" before the task
# agent even started, unrelated to either generation's actual code changes).
#
# This is defense in depth, not the primary fix -- the primary fix is
# giving Docker a much larger default-address-pools range (see
# /etc/docker/daemon.json), so hitting the ceiling at all should be rare
# even without this running. But it stops orphaned networks from
# accumulating indefinitely regardless of pool size.
#
#   nohup scripts/docker_network_prune_loop.sh >> docker_network_prune_loop.log 2>&1 &

INTERVAL_S=$((15 * 60))

while true; do
    echo "=== $(date -Iseconds) ==="
    docker network prune -f
    sleep "$INTERVAL_S"
done
