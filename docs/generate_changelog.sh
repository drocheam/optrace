#!/bin/bash

out=" Version  |      Date      |                                 Description "
out+=$'\n'
out+="-------------------------------------------------------------------------------------------------------------------------------------"
echo "$out" > ./docs/source/development/changelog.txt

git tag -l --sort=-refname --format='  %(refname:short)   |   %(creatordate:short)   |   %(contents:subject) %(contents:body)' >> ./docs/source/development/changelog.txt

