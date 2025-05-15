Here to run the local pipeline which should do the same thing as the webserver version (without the printed outputs and what not)

Go to the ./bashscripts/ folder

then do

sh pipeline_local.sh -f /abs/path/to/file -j NAME_OF_JOB -m MODEL_NAME -t THRESHOLD_TYPE -v THRESHOLD_VALUE -n N_POINTS -k N_JOBS

Usage: sh pipeline_local.sh -f <INPUTFILE> -j <JOBID> -m <MODEL> -t <THRESHOLD_TYPE> -v <THRESHOLD_VALUE> -n <n_points> -k <N_JOBS>

-m <MODEL> to choose between OSNOTRP, OSCSTRP, TSNOTRP, TSCSTRP

-t <THRESHOLD_TYPE> to choose between 'custom' or 'None'

-v <THRESHOLD_VALUE> is the actual distance threshold, in float between [0.01, 1.5] that you should input in case you used 'custom' threshold type

-n <N_POINTS> is the number of different thresholds to do in the optimization, 300 by default

-k <N_JOBS> is the number of cores to use in the clustering optimisation process, 20 by default