#! /usr/bin/bash
pepfile=${1}
basename=$(basename ${pepfile})
walltime=${2}
script_content=$(cat <<EOF
/home/projects/vaccine/people/morni/bin/pep2score_db_kernel -pa -t 2 ${pepfile} ${pepfile} > "/home/projects/vaccine/people/yatwan/tclustr/output/${basename}_all_sim.out"
EOF
)
echo "$script_content" > "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/TMP_tcr_sim.sh"
chmod +x "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/TMP_tcr_sim.sh"
qsub -W group_list=vaccine -A vaccine -m e -l nodes=1:ppn=40:thinnode,mem=36gb,walltime=${2} "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/TMP_tcr_sim.sh"
rm "/home/projects/vaccine/people/yatwan/tclustr/bashscripts/TMP_tcr_sim.sh"
