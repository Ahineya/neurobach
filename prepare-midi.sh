#!/bin/bash

rm -rf ./output/*.txt

cd midi

i=0

for f in ./*.mid; do
    echo $i
    ((i=i+1))
    echo "$f"

    python3 ../prepare.py --file "$f" > "../output/$f.txt" || echo "Error processing file"
done

cd ../output

cat *.txt > output.txt
