#!/bin/bash

history=$(git tag -n | sort -r)
current=$(git describe)

out=$(printf 'Current Version: %s \n\n%s' "$current" "$history")

touch ./source/changelog.txt
echo "$out" > ./source/changelog.txt
