Here to run the local pipeline which should do the same thing as the webserver version (without the printed outputs and what not).
There are some custom settings, but in theory, PLEASE USE -t "None" and not custom unless you are running a custom distance threshold. If you have many datapoints in your dataset, don't use too large of a N_POINTS or it will be extremely slow

You have full write permissions to ./tmp/ so don't mess it up

The model names are as follows :

OSNOTRP : One stage, No Triplet
OSCSTRP : One stage, Cosine Triplet
TSNOTRP : Two stage, No Triplet
TSCSTRP : Two stage, Cosine Triplet

### RUNNING THE PROGRAM ###

Go to the ./bashscripts/ folder
PROVIDE A JOB ID !!

bash pipeline_local.sh -f /abs/path/to/file -j NAME_OF_JOB -m MODEL_NAME -t THRESHOLD_TYPE -v THRESHOLD_VALUE -n N_POINTS -k N_JOBS

Usage: bash pipeline_local.sh -f <INPUTFILE> -j <JOBID> -m <MODEL> -t <THRESHOLD_TYPE> -v <THRESHOLD_VALUE> -n <n_points> -k <N_JOBS>

-m <MODEL> to choose between OSNOTRP, OSCSTRP, TSNOTRP, TSCSTRP

-t <THRESHOLD_TYPE> to choose between 'custom' or 'None'

-v <THRESHOLD_VALUE> is the actual distance threshold, in float between [0.01, 1.5] that you should input in case you used 'custom' threshold type

-n <N_POINTS> is the number of different thresholds to do in the optimization, 300 by default

-k <N_JOBS> is the number of cores to use in the clustering optimisation process, 20 by default
