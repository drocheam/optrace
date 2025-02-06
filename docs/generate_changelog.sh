#!/bin/bash

# don't do in github actions, as the online repo typically does not have the full tag history
if [[ "$GITHUB_ACTIONS" == "true" ]]; then
    exit 0
fi

# create table header
out=" Version  |      Date      |                                 Description "
out+=$'\n'
out+="-------------------------------------------------------------------------------------------------------------------------------------"
echo "$out" > ./docs/source/development/changelog.txt

# use git tag to create a table of label, date and description
# use awk to pad the label, as its length can differ
git tag -l --sort=-creatordate --format='%(refname:short) %(creatordate:short) %(contents:subject) %(contents:body)'  |\
    awk '{printf "  %-7s |   %s   |  %s  \n", $1, $2, substr($0, length($1 $2)+3)}' >> ./docs/source/development/changelog.txt

