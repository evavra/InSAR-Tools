#!/bin/bash

# Send email from command line
# $1 - body text
# $2 - subject line
# $3 - email address

echo $1 | mail -s $2 $3


