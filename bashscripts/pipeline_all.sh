#!/bin/bash

# This the main TCRcluster-1.0 script. It acts as the full pipeline, doing the NetMHCpan, KernDist, PepX query, and Python script
# Yat, Dec 2024

###############################################################################
#               GENERAL SETTINGS: CUSTOMIZE TO YOUR SITE
###############################################################################

if [ -z "$TMP" ]; then
	export TMP=/scratch
fi

# Default values
THRESHOLD=None
T_VALUE="None"
MODEL="TSCSTRP"

# Parse arguments from the form submission
while (( $# > 0 )); do
   case $1 in
     "--jobid")
       shift
       JOBID=$1
       ;;
     "--model")
       shift
       MODEL=$1
       ;;
     "--threshold_type")
       shift
       THRESHOLD_TYPE=$1  # Capture the threshold type (None or custom)
       ;;
     "--t_value")
       shift
       T_VALUE=$1  # Capture the custom threshold value
       ;;
     "--infile")
       shift
       FILENAME=$1
       ;;
   esac
   shift
done

## TODO : This is for command-line script debugging ; Comment this and switch to form submission
#while getopts ":f:j:m:t:v:" opt; do
#  case ${opt} in
#    f )
#      FILENAME=$OPTARG
#      ;;
#    j )
#      JOBID=$OPTARG
#      ;;
#    m )
#      MODEL=$OPTARG
#      ;;
#    t )
#      THRESHOLD_TYPE=$OPTARG
#      ;;
#    v )
#      T_VALUE=$OPTARG
#      ;;
#    \? )
#      echo "Usage: $0 -f <INPUTFILE> -o <OUTPUTDIRECTORY> -c <CHAINS> (ex: A1 A2 A3 B1 B2 B3) -s <SERVER> (c2/htc) -l <LABELCOL> -e <EXTRACOLS> -i <INDEXCOL>"
#      exit 1
#      ;;
#    : )
#      echo "Invalid option: -$OPTARG requires an argument"
#      exit 1
#      ;;
#  esac
#done

# Handle threshold logic
if [[ "$THRESHOLD_TYPE" == "custom" ]]; then
    if [[ -z "$T_VALUE" || "$T_VALUE" == "None" ]]; then
      echo "Error: Custom threshold selected but no value provided for --t_value."
      exit 1
    else
        THRESHOLD=$T_VALUE  # Set the threshold to the custom value
    fi
elif [[ "$THRESHOLD_TYPE" == "None" ]]; then
    THRESHOLD=None  # Use the default "None"
else
    echo "Error: Unknown threshold type '$THRESHOLD_TYPE'."
    exit 1
fi

filename=$(basename ${FILENAME})
basenm="${filename%.*}"

# determine platform
UNIX="Linux"
AR="x86_64"

# WWWROOT of web server
WWWROOT=/var/www/html

# WWWpath to service
SERVICEPATH=/services/TCRcluster-1.0

# other settings
PLATFORM="${UNIX}_${AR}"
USERDIR="/tools/src/TCRcluster-1.0"
BASHDIR="${USERDIR}/bashscripts/"
SRCDIR="${USERDIR}/src/"
DATADIR="${USERDIR}/data/"

# Use this as TMP dir for the webserver
#TMP=${WWWROOT}${SERVICEPATH}/tmp/${JOBID}/
# TODO : THIS IS FOR COMMANDLINE DEBUG ONLY
TMP="${USERDIR}/tmp/${JOBID}/"
chmod 755 $TMP

# Make this
mkdir -p ${TMP}
chmod 755 $TMP
mkdir -p /tmp/${JOBID} # ??

cd ${SRCDIR}
# TODO : check the arguments and change everything accordingly
chmod 755 "/home/locals/tools/src/TCRcluster-1.0/src/"
chmod 755 $SRCDIR
# Call the Python script with the correctly set threshold
PYTHON="/home/ctools/opt/anaconda3_202105/bin/python3"
#PYTHON=/home/people/riwa/anaconda3/envs/cuda/bin/python3.11
# todo: DEBUG with -np 10, njob 5 ; when done, remove
echo "Starting python script in $(pwd)"
# Debugging (Optional: Print variables to check values)
echo "PYTHON: $PYTHON"
echo "JOBID: $JOBID"
echo "MODEL: $MODEL"
echo "FILENAME: $FILENAME"
echo "THRESHOLD_TYPE: $THRESHOLD_TYPE"
echo "THRESHOLD: $THRESHOLD"
echo "TMP: $TMP"
cat $FILENAME

$PYTHON run_pipeline.py -j ${JOBID} -f ${FILENAME} --model ${MODEL} --threshold ${THRESHOLD} --outdir "${TMP}" -np 60 -n_jobs 10 > "${TMP}pylogs" 2>&1