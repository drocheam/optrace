#!/bin/bash

connections=$(node ./tests/audit_connections.js)
count=$(echo "$connections" | wc -l)

if [ "$count" -ne 1 ]; then
  echo "ERROR: Found more than one connection. Connection count: $count" >&2
  printf '%s\n' "$connections" >&2
  exit 1
fi

exit 0
