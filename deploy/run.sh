#!/usr/bin/bash
export JOB_ONCALL="ads_content_understanding"
export JOB_DATA_PROJECT="megataxon"

# the command passed is the name of the bash script and any arguments in this case
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR && "$@"
