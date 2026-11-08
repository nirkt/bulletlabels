#!/usr/bin/env bash
# End-to-end check (~2 min): setup ok, one solo capture, labels, a class scheme. Exits non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d -p "${TMPDIR:-/tmp}" blsmoke.XXXX); trap 'rm -rf "$T"' EXIT
python3 bl.py doctor
python3 bl.py capture --skin GunSlingerDefault --angles 0 "$T/cap"
python3 bl.py label "$T/cap" "$T/ds" --motion
n=$(ls "$T/ds/labels" | wc -l); [ "$n" -gt 10 ] || { echo "only $n labelled frames"; exit 1; }
python3 bl.py classes "$T/ds" --mode brawler_team --masks
python3 - "$T/ds" <<'PY'
import json, glob, sys
fs = glob.glob(sys.argv[1] + '/labels/*.json'); inst = [i for f in fs for i in json.load(open(f))['instances']]
assert inst, 'no labels at all'
assert all(i['name'] and i['kind'] and i['team'] for i in inst), 'missing fields'
print('smoke ok: %d frames, %d labels, kinds %s' % (len(fs), len(inst), sorted({i['kind'] for i in inst})))
PY
