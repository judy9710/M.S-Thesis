#!/bin/bash

if [ -e "32_naiveLog" ]; then
    result=$(grep "Total Time" "32_naiveLog" | awk -F"Total Time" '{print $2}' | awk '{print $1}')
    if [ -n "$result" ]; then
        echo "The word after "Total Time" is : $result"
    fi
else
    echo "file '32_naiveLog' not found"
fi
