#!/bin/bash

# process sgm files
cat $1 | grep "seg id" | sed 's/<seg id="[0-9]\+">//g' | sed 's/<\/seg>//g' > $2


# process tsv
grep '\S' input | cut -f1 > output1
grep '\S' input | cut -f2 > output2