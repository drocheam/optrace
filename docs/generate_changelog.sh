#!/bin/bash

history=$(git tag -n | sort -r)
current=$(git describe)

out=$(printf 'Current Version: %s \n\n%s' "$current" "$history")

touch ./docs/source/changelog.txt
echo "$out" > ./docs/source/changelog.txt
