#! /usr/bin/bash

# HERE IT DEPENDS ON WHETHER WE ARE USING C2 OR HTC

# RID part
# Define the characters that can be used
characters="abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ0123456789"
# Generate a random index between 0 and 61 (total number of characters)
index=$((RANDOM % 60))
# Get the character at the generated index
first_char="${characters:index:1}"
# Generate the remaining 4 characters as a combination of the defined characters
rest_chars=$(head /dev/urandom | tr -dc "$characters" | head -c 4)
# Combine the first and remaining characters
random_id="${first_char}${rest_chars}"
# Datetime part
datetime_string=$(date +"%y%m%d_%H%M%S")
start_time=$(date +%s)
# Setting default paths
CHAINS=("A1" "A2" "A3" "B1" "B2" "B3")  # Default chains
LABELCOL="peptide"
INDEXCOL=None
EXTRACOLS=("original_peptide" "binder" "partition" "original_index")
TBCRALIGN="/home/projects/vaccine/people/morni/bin/tbcr_align"
HOMEDIR="/home/projects/vaccine/people/yatwan/tclustr/"
chainarg="full"
# Pre-set the name with a random id, then parse the -o if exists
OUTNAME="MST_${random_id}"
OUTPUTDIRECTORY=${OUTNAME}
PTFILE=""
JSONFILE=""
# HANDLE SINGLE LETTER ARGS

while getopts ":f:c:s:l:e:o:i:p:j:" opt; do
  case ${opt} in
    f )
      INPUTFILE=$OPTARG
      ;;
    o )
      OUTPUTDIRECTORY="${OPTARG}_${OUTPUTDIRECTORY}"
      echo "HERE ${OUTPUTDIRECTORY}"
      ;;
    c )
      # If -c is used, override the default chains
      CHAINS=("$OPTARG")  # Add the first option after -c
      while [[ ${!OPTIND} =~ ^[^-] ]]; do
        CHAINS+=("${!OPTIND}")
        OPTIND=$((OPTIND + 1))
      done
      ;;
    s )
      # Health Tech server
      if [ "$OPTARG" == "htc" ]; then
        SERVER="$OPTARG"
        CONDA=/home/people/riwa/anaconda3/etc/profile.d/conda.sh
        TBCRALIGN="/home/people/morni/bin/tbcr_align"
        HOMEDIR='/home/projects2/riwa/tclustr/'
      # Computerome
      elif [ "$OPTARG" == "c2" ]; then
        SERVER="$OPTARG"
        CONDA=/home/projects/vaccine/people/yatwan/anaconda3/etc/profile.d/conda.sh
        TBCRALIGN="/home/projects/vaccine/people/morni/bin/tbcr_align"
        HOMEDIR="/home/projects/vaccine/people/yatwan/tclustr/"
      fi
      ;;
    e )
      EXTRACOLS=("$OPTARG")  # Add the first option after -c
      while [[ ${!OPTIND} =~ ^[^-] ]]; do
        EXTRACOLS+=("${!OPTIND}")
        OPTIND=$((OPTIND + 1))
      done
      ;;
    l )
      LABELCOL="$OPTARG"  # Add the first option after -c
      ;;
    i )
      INDEXCOL="$OPTARG"
      ;;
    p )
      PTFILE="$OPTARG"
      ;;
    j )
      JSONFILE="$OPTARG"
      ;;
    \? )
      echo "Usage: $0 -f <INPUTFILE> -o <OUTPUTDIRECTORY> -c <CHAINS> (ex: A1 A2 A3 B1 B2 B3) -s <SERVER> (c2/htc) -l <LABELCOL> -e <EXTRACOLS> -i <INDEXCOL> -p <PTFILE> -j <JSONFILE>"
      exit 1
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument"
      exit 1
      ;;
  esac
done


# Then add the datetime
OUTPUTDIRECTORY="${datetime_string}_${OUTPUTDIRECTORY}"
# Get the full paths depending on the server used
PYDIR="${HOMEDIR}pyscripts/"
BASHDIR="${HOMEDIR}bashscripts/"

# that's where all the outputs will be saved. We just need "OUTPUTDIRECTORY"
# because all the other scripts handle the "../output/${OUTPUTDIRECTORY}" by default
OUTDIR="$(realpath "$(pwd)/../output/${OUTPUTDIRECTORY}/")/"
mkdir -pv $OUTDIR

source ${CONDA}
source activate tcrdist3
cd $BASHDIR
sh do_tbcralign.sh -f $INPUTFILE -o ${OUTPUTDIRECTORY} -c "${CHAINS[@]}" -s $SERVER -l $LABELCOL -e "${EXTRACOLS[@]}"

cd $PYDIR
python3 do_tcrdist.py -f $INPUTFILE -od ${OUTPUTDIRECTORY} -pep $LABELCOL -others "${EXTRACOLS[@]}" -idx $INDEXCOL


tbcrfile="$(ls $OUTDIR*TBCR_distmatrix*.csv)"
tcrdistfile="$(ls $OUTDIR*tcrdist3_distmatrix*.txt)"

echo "#######################################"
echo ""
echo "Running script using the following:"
echo "#######################################"
echo ""
echo "tbcrfile: ${tbcrfile}"
echo "tcrdistfile: ${tcrdistfile}"
echo "PTFILE: ${PTFILE}"
echo "JSON: ${JSONFILE}"
echo "#######################################"
echo ""
source ${CONDA}
conda activate cuda

#sys exit 1
python3 240819_MST_cuts_clustering.py -pt_file ${PTFILE} -json_file ${JSONFILE} -f $INPUTFILE -tcrdist $tcrdistfile -tbcralign $tbcrfile -od ${OUTPUTDIRECTORY} -rid 'clstr' -index_col $INDEXCOL -rest_cols "${EXTRACOLS[@]}" -label_col ${LABELCOL} -n_jobs 40

endtime=$(date +"%y%m%d_%H%M%S")
# Record the end time
end_time=$(date +%s)

# Calculate the duration in seconds
duration=$(( end_time - start_time ))

# Convert the duration to HH:mm:ss format
hours=$(( duration / 3600 ))
minutes=$(( (duration % 3600) / 60 ))
seconds=$(( duration % 60 ))

# Format the output to HH:mm:ss
printf -v elapsed_time "%02d:%02d:%02d" $hours $minutes $seconds

# Output the elapsed time
echo "Time taken: $elapsed_time"
