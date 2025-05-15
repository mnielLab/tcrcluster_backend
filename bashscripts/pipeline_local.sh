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
N_VALUE=300

# TODO : This is for command-line script debugging ; Comment this and switch to form submission
while getopts ":f:j:m:t:v:n:" opt; do
  case ${opt} in
    f )
      FILENAME=$OPTARG
      ;;
    j )
      JOBID=$OPTARG
      ;;
    m )
      # This should be OSNOTRP, OSCSTRP, TSNOTRP, TSCSTRP
      MODEL=$OPTARG
      ;;
    t )
      # This is "custom" or "None"
      THRESHOLD_TYPE=$OPTARG
      ;;
    v )
      # This inputs a single distance threshold
      T_VALUE=$OPTARG
      ;;
    n )
      # This gives the number of points (number of thresholds to try) to optimize
      N_VALUE=$OPTARG
    ;;
    \? )
      echo "Usage: $0 -f <INPUTFILE> -j <JOBID> -m <MODEL> -t <THRESHOLD_TYPE> -v <THRESHOLD_VALUE> -n <n_points>"
      exit 1
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument"
      exit 1
      ;;
  esac
done

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


PLATFORM="${UNIX}_${AR}"
USERDIR="/home/projects2/riwa/tcrcluster_backend/"
BASHDIR="${USERDIR}/bashscripts/"
SRCDIR="${USERDIR}/src/"
DATADIR="${USERDIR}/data/"

# Use this as TMP dir for the webserver
TMP=${USERDIR}/tmp/${JOBID}/
# THIS IS FOR COMMANDLINE DEBUG ONLY
#TMP="${USERDIR}/tmp/${JOBID}/"
chmod 755 $TMP

# Make this
mkdir -p ${TMP}
chmod 777 $TMP
#mkdir -p /tmp/${JOBID} # ??

cd ${SRCDIR}
chmod 755 "/home/locals/tools/src/TCRcluster-1.0/src/"
chmod 755 $SRCDIR
# Call the Python script with the correctly set threshold
PYTHON="/home/ctools/opt/anaconda3_202105/bin/python3"
#PYTHON=/home/people/riwa/anaconda3/envs/cuda/bin/python3.11
# todo: DEBUG with -np 10, njob 5 ; when done, remove
echo "JOBID: $JOBID"
echo "MODEL: $MODEL"
echo "FILENAME: $FILENAME"
echo "THRESHOLD_TYPE: $THRESHOLD_TYPE"
echo "THRESHOLD: $THRESHOLD"

$PYTHON run_pipeline.py -j ${JOBID} -f ${FILENAME} --model ${MODEL} --threshold ${THRESHOLD} --outdir "${TMP}" -np $n -n_jobs 20 > "${TMP}pylogs.log" 2>&1