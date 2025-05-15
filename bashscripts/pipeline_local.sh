#!/bin/bash

# This the main TCRcluster-1.0 script.
# To make this work on your local machine, REPLACE THE USERDIR VARIABLE TO WHERE YOU HAVE TCRCLUSTER INSTALLED
# Yat, May 2025

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
N_JOBS=20
# TODO : This is for command-line script debugging ; Comment this and switch to form submission
while getopts ":f:j:m:t:v:n:k:" opt; do
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
    k )
      # This gives the NUMBER OF PARALLEL JOBS FOR THE CLUSTERING
      N_JOBS=$OPTARG
    ;;
    \? )
      echo "Usage: $0 -f <INPUTFILE> -j <JOBID> -m <MODEL> -t <THRESHOLD_TYPE> -v <THRESHOLD_VALUE> -n <n_points> -k <N_JOBS>\n-m<MODEL> to choose between OSNOTRP, OSCSTRP, TSNOTRP, TSCSTRP; -t <THRESHOLD_TYPE> to choose between 'custom' or 'None'; -v <THRESHOLD_VALUE> is the actual distance threshold, in float between [0.01, 1.5] that you should input in case you used 'custom' threshold type; -n <N_POINTS> is the number of different thresholds to do in the optimization, 300 by default; -k <N_JOBS> is the number of cores to use in the clustering optimisation process, 20 by default"

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


USERDIR="/home/projects2/riwa/tcrcluster_backend/"
BASHDIR="${USERDIR}/bashscripts/"
SRCDIR="${USERDIR}/src/"
DATADIR="${USERDIR}/data/"

# Use this as TMP dir for the webserver
TMP=${USERDIR}tmp/${JOBID}/

# Make this
mkdir -p ${TMP}

cd ${SRCDIR}

# Call the Python script with the correctly set threshold
PYTHON="/home/ctools/opt/anaconda3_202105/bin/python3"
#PYTHON=/home/people/riwa/anaconda3/envs/cuda/bin/python3.11
# todo: DEBUG with -np 10, njob 5 ; when done, remove
echo "JOBID: $JOBID"
echo "MODEL: $MODEL"
echo "FILENAME: $FILENAME"
echo "THRESHOLD_TYPE: $THRESHOLD_TYPE"
echo "THRESHOLD: $THRESHOLD"

$PYTHON pipeline_local.py -j ${JOBID} -f ${FILENAME} --model ${MODEL} --threshold ${THRESHOLD} --outdir "${TMP}" -np $N_VALUE -n_jobs $N_JOBS > "${TMP}pylogs.log" 2>&1
