#!/bin/bash

# don't do in github actions, as the online repo typically does not have the full tag history
if [[ "$GITHUB_ACTIONS" == "true" ]]; then
    exit 0
fi

# create table header
# table should have a width of 100
printf " Version  |      Date      |                             Description
----------|----------------|------------------------------------------------------------------------\n" \
> ./docs/source/development/changelog.txt

# use git tag to create a table of label, date and description
# use grep to only include version tags such as 1.7.10, not the release tags such as 1.7.10-c45a756
# create a table with wrapped description field
git tag -l --sort=-creatordate --format='%(refname:short) %(creatordate:short) %(contents:subject) %(contents:body)' |\
grep '[0-9]\+\.[0-9]\+\.[0-9]\+ ' | \
while IFS= read -r line; do
    # Extract the first two columns (version and date)
    version=$(echo "$line" | awk '{print $1}')
    date=$(echo "$line" | awk '{print $2}')

    # Extract the description
    description=$(echo "$line" | awk '{print substr($0, length($1 $2)+3)}')

    # wrap description using fold. -s option breaks at spaces
    wrapped_description=$(echo "$description" | fold -s -w 71)

    # Read the wrapped description line by line
    first_line=true
    while IFS= read -r desc_line; do
        if [ "$first_line" = true ]; then
            # Print the first line with the version and date
            printf "  %-7s |   %s   | %s\n" "$version" "$date" "$desc_line"
            first_line=false
        else
            # Print subsequent wrapped lines
            printf "          |                | %s\n" "$desc_line"
        fi
    done <<< "$wrapped_description"
    echo "----------|----------------|------------------------------------------------------------------------"

done >> ./docs/source/development/changelog.txt 
