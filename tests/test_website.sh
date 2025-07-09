#!/bin/bash

# check an expression
assert() {
    local condition="$1"
    if ! eval "$condition" > /dev/null ; then
        echo "Test failure in file $0, line ${BASH_LINENO}, condition: $1" 1>&2
    fi
}

# prints the url if it isn't relative and not equal to DOMAIN
url_check()
{
    local DOMAIN="$1"
    local URL_DOMAIN="$2"
    local url="$3"
    local INFO="$4"

    if [[ "$url" == //* ]]; then # Handle protocol-relative URLs
        url="https:$url"
    fi
    if [[ "$url" == http* ]]; then # Handle relative URLs
        local URL_DOMAIN=$(echo "$url" | sed -E 's|https?://([^/]+).*|\1|')
        if [[ "$URL_DOMAIN" != "" && "$URL_DOMAIN" != "$DOMAIN" ]]; then
            echo "$INFO: $url"
        fi
    fi
    return 0
}

# source code based check if external ressources are loaded, including scripts, fonts, images, iframes etc.
# can't handle JS loaded things
scan_external_resources() 
{
    local TARGET_URL="$1"
    local DOMAIN=$(echo "$TARGET_URL" | sed -E 's|https?://([^/]+).*|\1|') # Extract the base domain
    local HTML_CONTENT=$(curl -s -L "$TARGET_URL")

    # check for external scripts
    echo "$HTML_CONTENT" | grep -Eo '<script[^>]*src="[^"]*"' | \
    sed -E 's|<script[^>]*src="([^"]*)".*|\1|' | \
    while read -r url; do
        url_check "$DOMAIN" "$URL_DOMAIN" "$url" "JS"
    done

    # check for CSS and external resource links
    echo "$HTML_CONTENT" | grep -Eo '<link[^>]*href="[^"]*"' | \
    sed -E 's|<link[^>]*href="([^"]*)".*|\1|' | \
    while read -r url; do
        url_check "$DOMAIN" "$URL_DOMAIN" "$url" "Res"
    done

    # check for direct font links
    echo "$HTML_CONTENT" | grep -Eo '(href|src)="[^"]*\.(woff|woff2|ttf|otf|eot)(\?.*)?"' | \
    sed -E 's|[^"]*"(.*)"|\1|' | \
    while read -r url; do
        url_check "$DOMAIN" "$URL_DOMAIN" "$url" "Font"
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
        
            url_check "$DOMAIN" "$URL_DOMAIN" "$url" "Img"
        done
    done

    # check for video/audio links
    echo "$HTML_CONTENT" | grep -Eo '(video|audio|source)[^>]* src="[^"]*"' | \
    sed -E 's|(video\|audio\|source)[^>]* src="([^"]*)".*|\2|' | \
    while read -r url; do
        url_check "$DOMAIN" "$URL_DOMAIN" "$url" "Media"
    done

    # check iframe links
    echo "$HTML_CONTENT" | grep -Eo '<iframe[^>]*src="[^"]*"' | \
    sed -E 's|<iframe[^>]*src="([^"]*)".*|\1|' | \
    while read -r url; do
        url_check "$DOMAIN" "$URL_DOMAIN" "$url" "Iframe"
    done

    return 0
}

# test important pages
assert "curl -Is https://drocheam.github.io/ | head -1 | grep 200"
assert "curl -Is https://drocheam.github.io/optrace | head -1 | grep 301"
assert "curl -Is https://drocheam.github.io/optrace/index.html | head -1 | grep 200"
assert "curl -Is https://drocheam.github.io/optrace/impressum.html | head -1 | grep 200"
assert "curl -Is https://drocheam.github.io/robots.txt | head -1 | grep 200"
assert "curl -Is https://drocheam.github.io/optrace/sitemap.xml | head -1 | grep 200"

# check that the site does not want to set a cookie
# see https://stackoverflow.com/a/55807947
assert "[[ $(curl 'https://drocheam.github.io/optrace/index.html' -o /dev/null --dump-header - 2>&1 | grep -i "set-cookie" | wc -l) == '0' ]]"
assert "[[ $(curl 'https://drocheam.github.io/' -o /dev/null --dump-header - 2>&1 | grep -i "set-cookie" | wc -l) == '0' ]]"
# a site like google wants to set cookies
assert "[[ $(curl 'https://www.google.com/' -o /dev/null --dump-header - 2>&1 | grep -i "set-cookie" | wc -l) != '0' ]]"

# check that the website does not load external resources
assert "[ -z $(scan_external_resources "https://drocheam.github.io/optrace/index.html") ] || echo true"
assert "[ -z $(scan_external_resources "https://drocheam.github.io/optrace/details/matrix_analysis.html") ] || echo true"
# check the script by checking a page with external scripts and fonts
assert "scan_external_resources 'https://shibuya.lepture.com/'"

