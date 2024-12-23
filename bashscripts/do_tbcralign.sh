#! /usr/bin/bash
# Here either /home/projects/vaccine/ or /home/ depending on computerome or HTC
start_time=$(date +%s)

C2PATH="/home/projects/vaccine/people/morni/bin/tbcr_align"
HTCPATH="/home/people/morni/bin/tbcr_align"
CHAINS=("A1" "A2" "A3" "B1" "B2" "B3")  # Default chains
LABELCOL="peptide"
EXTRACOLS=("original_peptide" "binder" "partition" "original_index")
TBCRALIGN="$HTCPATH"
chainarg="full"
OUTPUTDIRECTORY=""
# args : 1 = mainfolder ; 2 = outfolder ; 3 = n_epochs ; 4 = grep
while getopts ":f:c:s:l:e:o:" opt; do
  case ${opt} in
    f )
      INPUTFILE=$OPTARG
      ;;
    o )
      OUTPUTDIRECTORY="$OPTARG"
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
      if [ "$OPTARG" == "htc" ]; then
        TBCRALIGN="$HTCPATH"
      elif [ "$OPTARG" == "c2" ]; then
        TBCRALIGN="$C2PATH"
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
    \? )
      echo "Usage: $0 -f <INPUTFILE> -o <OUTPUTDIRECTORY> -c <CHAINS> (ex: A1 A2 A3 B1 B2 B3) -s <SERVER> (c2/htc) -l <LABELCOL> -e <EXTRACOLS>"
      exit 1
      ;;
    : )
      echo "Invalid option: -$OPTARG requires an argument"
      exit 1
      ;;
  esac
done

# Shift the processed options so that $1, $2, etc. now refer to non-option arguments
shift $((OPTIND - 1))

OUTDIR="$(realpath "$(pwd)/../output/${OUTPUTDIRECTORY}/")/"
mkdir -pv $OUTDIR
# Extract the basename without the extension using parameter expansion
# Here assume we use the full TCRs
filename=$(basename "$INPUTFILE")
basename_without_extension="${filename%.*}"
tmppath="${OUTDIR}${basename_without_extension}_TMP.txt" # Saves the intermediate tsv file for TBCRalign
tbcrtmp="${OUTDIR}${basename_without_extension}_TBCRraw_TMP.txt" # Saves the output of the TBCRalign to be read to convert to distmatrix
output_name="${OUTDIR}${basename_without_extension}_TBCR_distmatrix.csv" # final output name to save the dm
string_chains=$(printf '%s' "$(IFS=','; echo "\"${CHAINS[*]}\"")")
string_extracols=$(printf '%s' "$(IFS=','; echo "\"${EXTRACOLS[*]}\"")")
# echo the variables to check stuff
# FILES MUST BE IN SAVED IN THE RIGHT FORMAT ; see

  # Embedded python code to save the df in the required format
python3 -W ignore <<EOF
print('Saving tmp file')
import pandas as pd
filepath = "${INPUTFILE}"
tmppath = "${tmppath}"
chains = eval('${string_chains}').split(',')
df = pd.read_csv(filepath)
c='binder' if 'binder' in df.columns else 'target' if 'target' in df.columns else None
if c is None:
  c='binder'
  df[c] = 0.5 # Create a fake column
df[chains+[c]].to_csv(tmppath, sep='\t', header=False)
EOF
# Run TBCRalign
echo "Running TBCR"
$TBCRALIGN -a -w 1,1,4,1,1,4 $tmppath > $tbcrtmp
# Embedded Python code to recover the distance matrix
python3 -W ignore <<EOF
print('Reading TBCRalign output and converting to dist_matrix')
import pandas as pd
import numpy as np
# Accessing the Bash variable in Python
output_filename = "${output_name}"
print('Read TBCR raw')
tbcr = pd.read_csv("${tbcrtmp}", comment='#', sep =' ', header=None, index_col=None)
chains = eval('${string_chains}').split(',')
extra_cols = eval('${string_extracols}').split(',')
extra_cols = [x for x in extra_cols if len(x)>0] # bad fix for my optargs handling
qcols = [f'q_{x}' for x in chains + ['binder']]
dbcols = [f'db_{x}' for x in chains]
print('Setting columns ;')
# Replace col names
tbcr.columns = ['kind', 'source', 'q_index'] + qcols + ['target', 'db_index'] + dbcols + ['score'] + ['db_binder']
print('Reset columns')
# Create a copy and switch around col names (To create lower triangle of the square matrix)
print('Creating copy and swapping cols')
fake_tbcr = tbcr.copy()
fake_tbcr.columns = ['kind', 'name', 'db_index']+ dbcols + ['db_binder', 'target','q_index'] + [x for x in qcols if not x.endswith('binder')] + ['score','q_binder']

print('Concat')
# concatenate and create a new df with only the indices to make it lighter
tbcr_cat = pd.concat([tbcr, fake_tbcr])

print('Getting idx_scores')
# Keeping only index and score as columns
tbcr_idx_scores = tbcr_cat.drop(columns=[x for x in tbcr_cat.columns if 'index' not in x and 'score' not in x])

print('Creating idx df')
# Create a new index to TCR df to re-index later
index_tcrs = tbcr_cat.drop_duplicates(subset=['q_index'])
index_tcrs['tcr'] = index_tcrs[[x for x in qcols if x != 'q_binder' and x != 'q_label' and x !='q_target']].sum(axis=1)
index_tcrs = index_tcrs[['q_index', 'tcr']]

print('Adding self scores and pivoting')
# Add the self scores (12) to the final df before pivoting
tbcr_idx_scores = pd.concat([tbcr_idx_scores, pd.DataFrame([[x, x, 12.] for x in tbcr_idx_scores.q_index. unique()], columns=['q_index', 'db_index', 'score'])])

# Pivot and do 1 - x / 12 to get a square distance matrix instead of a similarity matrix with max value 12
dist_matrix = 1-(tbcr_idx_scores.pivot_table(index='q_index', columns='db_index', values='score')/12)


# TODO: Get filename and get chains variable in here
original_filename = "${INPUTFILE}"
original_df = pd.read_csv(original_filename)
original_df['tcr'] = original_df[chains].sum(axis=1)
len_before = len(original_df)
original_df = original_df.merge(index_tcrs[['q_index','tcr']], left_on='tcr', right_on='tcr').drop_duplicates(['q_index','tcr'])
len_after = len(original_df)
extra_cols = [x for x in extra_cols if x in original_df.columns]
print(f'Saving dist_matrix with extra columns {extra_cols}')
if len_before!=len_after:
	print(f'things went wrong, before = {len_before}, after = {len_after}')
dist_matrix = dist_matrix.merge(original_df.set_index('q_index')[extra_cols], left_index=True, right_index=True)
dist_matrix.to_csv(output_filename)
print(f'TBCR distmatrix saved at {output_filename}')
EOF

rm ${OUTDIR}*TMP*.txt


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
echo "TBCRalign : Time taken: $elapsed_time"

