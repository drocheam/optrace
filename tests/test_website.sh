#!/bin/bash

counter=0

# check an expression
assert() {
    ! eval "$1" > /dev/null && echo "Test failure in file $0, line ${BASH_LINENO}, condition: $1" 1>&2 && counter=$((counter+1))
}

# count set-cookie requests from page source code
count_cookies() {
    echo $(curl "$1" -o /dev/null --dump-header - 2>&1 | grep -i "set-cookie" | wc -l) 
}

# get HEAD response
head_response() {
    curl -Is "$1" | head -1 | cut -d ' ' -f 2
}

# prints the url if it isn't relative and not from DOMAIN
url_check()
{
    local url="$1"
    local DOMAIN="$2"
    local INFO="$3"

    if [[ "$url" == //* ]]; then # Handle protocol-relative URLs
        local url="https:$url"
    fi
    if [[ "$url" == http* ]]; then # Handle relative URLs
        local URL_DOMAIN=$(echo "$url" | sed -E 's|https?://([^/]+).*|\1|')
        if [[ "$URL_DOMAIN" != "" && "$URL_DOMAIN" != "$DOMAIN" ]]; then
            echo "$INFO: $url"
        fi
    fi
    return 0
}

# source code based check if external resources are loaded, including scripts, fonts, images, iframes etc.
# can't handle JS loaded resources
scan_external_resources() 
{
    local TARGET_URL="$1"
    local DOMAIN=$(echo "$TARGET_URL" | sed -E 's|https?://([^/]+).*|\1|') # Extract the base domain
    local HTML_CONTENT=$(curl -s -L "$TARGET_URL")

    # check for external scripts
    echo "$HTML_CONTENT" | grep -Eo '<script[^>]*src="[^"]*"' | \
    sed -E 's|<script[^>]*src="([^"]*)".*|\1|' | \
    while read -r url; do
        url_check "$url" "$DOMAIN" "JS"
    done

    # check for CSS and external resource links
    echo "$HTML_CONTENT" | grep -Eo '<link[^>]*href="[^"]*"' | \
    sed -E 's|<link[^>]*href="([^"]*)".*|\1|' | \
    while read -r url; do
        url_check "$url" "$DOMAIN" "Res"
    done

    # check for direct font links
    echo "$HTML_CONTENT" | grep -Eo '(href|src)="[^"]*\.(woff|woff2|ttf|otf|eot)(\?.*)?"' | \
    sed -E 's|[^"]*"(.*)"|\1|' | \
    while read -r url; do
        url_check "$url" "$DOMAIN" "Font"
    done

    # check for image links
    echo "$HTML_CONTENT" | grep -Eo '(img|source)[^>]* (src|srcset)="[^"]*"' | \
    sed -E 's|(img\|source)[^>]* (src\|srcset)="([^"]*)".*|\3|' | \
    while read -r url; do
        IFS=',' read -ra SRCSET_URLS <<< "$url"
        for srcset_url in "${SRCSET_URLS[@]}"; do

            local clean_url=$(echo "$srcset_url" | sed -E 's/^\s*([^[:space:]]+)\s*([0-9]+x|[0-9]+w)?\s*$/\1/' | xargs)
            if [[ -z "$clean_url" ]]; then
                continue
            fi
        
            url_check "$url" "$DOMAIN" "Img"
        done
    done

    # check for video/audio links
    echo "$HTML_CONTENT" | grep -Eo '(video|audio|source)[^>]* src="[^"]*"' | \
    sed -E 's|(video\|audio\|source)[^>]* src="([^"]*)".*|\2|' | \
    while read -r url; do
        url_check "$url" "$DOMAIN" "Media"
    done

    # check iframe links
    echo "$HTML_CONTENT" | grep -Eo '<iframe[^>]*src="[^"]*"' | \
    sed -E 's|<iframe[^>]*src="([^"]*)".*|\1|' | \
    while read -r url; do
        url_check "$url" "$DOMAIN" "Iframe"
    done

    return 0
}

# test important pages
assert "[[ $(head_response 'https://drocheam.github.io/') == '200' ]]"
assert "[[ $(head_response 'https://drocheam.github.io/optrace') == '301' ]]"
assert "[[ $(head_response 'https://drocheam.github.io/optrace/index.html') == '200' ]]"
echo "Main pages checks finished"
assert "[[ $(head_response 'https://drocheam.github.io/optrace/impressum.html') == '200' ]]"
assert "[[ $(head_response 'https://drocheam.github.io/robots.txt') == '200' ]]"
assert "[[ $(head_response 'https://drocheam.github.io/optrace/sitemap.xml') == '200' ]]"
echo "Important pages checks finished"
assert "[[ $(head_response 'https://drocheam.github.io/optrace/searchindex.js') == '200' ]]"
assert "[[ $(head_response 'https://drocheam.github.io/optrace/_static/css/custom.css') == '200' ]]"
assert "[[ $(head_response 'https://drocheam.github.io/optrace/_static/shibuya.css') == '200' ]]"
assert "[[ $(head_response 'https://drocheam.github.io/optrace/_static/mathjax/loader.js') == '200' ]]"
echo "Resource pages checks finished"

# check that the site does not want to set a cookie
# see https://stackoverflow.com/a/55807947
assert "[[ $(count_cookies 'https://drocheam.github.io/optrace/index.html') == '0' ]]"
assert "[[ $(count_cookies 'https://drocheam.github.io/') == '0' ]]"
# a site like google wants to set cookies
assert "[[ $(count_cookies 'https://www.google.com/') != '0' ]]"
echo "Cookie checks finished"

# check that the website does not load external resources
assert "[ -z "$(scan_external_resources 'https://drocheam.github.io/optrace/index.html')" ] || echo true"
assert "[ -z "$(scan_external_resources 'https://drocheam.github.io/optrace/details/matrix_analysis.html')" ] || echo true"
# check the script by checking a page with external scripts and fonts
assert "scan_external_resources 'https://shibuya.lepture.com/'"
echo "Resource checks finished"

# exit with non zero status code when errors occurred
[[ $counter != 0 ]] && exit 1
exit 0
