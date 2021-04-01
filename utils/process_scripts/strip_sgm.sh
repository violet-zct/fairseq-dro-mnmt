#!/bin/bash

# process tmx files
sed -n 's:.*<seg>\(.*\)</seg>.*:\1:p' < en-tr.tmx > output

# odd numbered lines
sed -n 'n;p' < output > wmt18.en-tr.en

# even numbered lines
sed -n 'p;n' < output > wmt18.en-tr.tr


# process sgm files
