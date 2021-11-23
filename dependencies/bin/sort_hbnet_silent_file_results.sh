#! /bin/bash
# from a silent file provided as argument 1 ($1), grab the top X structures integer from argument 2 ($2),
# using the shape_complementarity and energy_density overlap, and finally the structures
# which have the highest residue count from this overlap

# create the file which contains the hbnet scores
hbnet_scores=$(dirname -- "$1")/hbnet.sc  # $(basename -- "$1")}
grep "^SCORE" $1 > $hbnet_scores


# find the column number from the scores file
decoy=$(awk -v field=description 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
hbnet_sc=$(awk -v field=R_shape_complementarity_hbnet_core 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
hbnet_density=$(awk -v field=R_interaction_energy_density 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
hbnet_res_count=$(awk -v field=R_core_design_residue_count 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
# Symmetrized descriptors
# hbnet_sc=$(awk -v field=sc_hbnet_core 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
# hbnet_density=$(awk -v field=ie_density_hbnet_core 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
# hbnet_res_count=$(awk -v field=residue_count_hbnet_core 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
# hbnet_density=$(awk -v field=score 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)								# test
# hbnet_res_count=$(awk -v field=fa_rep 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)								# test


# sort -k"$khbnet_sc"n -k"$khbnet_density"n $hbnet_scores	# nr sorts each key (k) numerically (n) in descending (r) order->Higher at the top	# test
# sort the scores by the respective column numbers identified, then extract the decoy name
# only sorts the first example
sc_array=($(tail -n +2 $hbnet_scores | sort -u -k"$decoy"r | sort -k"$hbnet_sc"nr | head -n $(($2 * 3)) | awk -v field=$decoy '{print $field}'))
density_array=($(tail -n +2 $hbnet_scores | sort -u -k"$decoy"r | sort -k"$hbnet_density"n | head -n $(($2 * 3)) | awk -v field=$decoy '{print $field}'))
residue_count_array=($(tail -n +2 $hbnet_scores | sort -u -k"$decoy"r | sort -k"$hbnet_res_count"nr | awk -v field=$decoy '{print $field}'))


# sort the elements in both density and sc (shape complementarity) array by the sc order adding to "overlap"
regex_string=" ${density_array[*]} "            # add framing blanks
for item in ${sc_array[@]}; do
  if [[ $regex_string =~ " $item " ]] ; then    # use $item as regexp
    overlap+=($item)
  fi
done
# in case the overlap is shorter than the requested amount in $2, add all the members from the sc array
if [[ ${#overlap[@]} < $2 ]] ; then
  for ((i=0 ; i<$2 ; i++)); do
	overlap+=${sc_array[$i]}
  done
fi
# echo  ${overlap[@]}  																															# test


# sort the overlapped array by the number of residues involved, exhausting all decoys which overlapped above 
regex_string=" ${overlap[*]} "
for item in ${residue_count_array[@]}; do
  if [[ $regex_string =~ " $item " ]] ; then
    final_overlap+=($item)
  fi
done
# echo  ${final_overlap[@]}  																													# test

# create the files where the information will be stored
hbnet_tags_to_grab=$(dirname -- "$1")/hbnet_selected.tags
if [ -f $hbnet_tags_to_grab ]; then
  rm $hbnet_tags_to_grab
fi
final_silent_file=$(dirname -- "$1")/hbnet_selected.o
#if [ -f $final_silent_file ]; then
#  rm $final_silent_file
#fi
grep "^SEQUENCE" --max-count=1 $1 > $final_silent_file
grep "^SCORE" --max-count=1 $1 >> $final_silent_file
echo "REMARK BINARY SILENTFILE" >> $final_silent_file

# finally, create the decoys of interest adding each decoy until the specified number provided in $2 is reached
if [ ${#final_overlap[@]} -lt $2 ]; then
    max_length=${#final_overlap[@]}
  else
    max_length=$2
fi
for ((i=0 ; i<$max_length ; i++)); do
  echo ${final_overlap[$i]} >> $hbnet_tags_to_grab
  structure_line_numbers=($(grep "${final_overlap[$i]}" -n $1 | cut --fields=1 --delimiter=:))
  duplicate_structure=0
  last_array_num=$((${structure_line_numbers[0]} - 1))  # always satisfies initial condition
  for ((array_num=0 ; array_num<${#structure_line_numbers[@]} ; array_num++)); do
    # is the next line way different than the last and 100 away from line start?
    # assumes that the silent file has at least 100 lines present thus allows 99 REMARK lines to be read (current investigation shows >300 lines usually)
    if [ $((${structure_line_numbers[$array_num]} - 1)) != $last_array_num ] && [ $array_num -gt 100 ]; then
#      echo "The array number is $array_num which gives the value $((${structure_line_numbers[$array_num]})) and the last array number was $last_array_num"
#      [ $array_num -lt 100 ] && echo "all is good"
      duplicate_structure=$last_array_num  #${structure_line_numbers[$array_num]}
      break
    fi
    last_array_num=${structure_line_numbers[$array_num]}
  done
  if [ $duplicate_structure == 0 ]; then  # not a duplicate structure
    head -n ${structure_line_numbers[-1]} $1 | tail -$((${structure_line_numbers[-1]} - ${structure_line_numbers[0]} + 1)) >> $final_silent_file
    else  # only take the first example
      echo "Found a duplicate structure starting at line $duplicate_structure taking only the first occurrence"
      head -n $duplicate_structure $1 | tail -$(($duplicate_structure - ${structure_line_numbers[0]} + 1)) >> $final_silent_file  # | awk -v decoy=${final_overlap[$i]}_0001 '{print $1, decoy}' >> $final_silent_file
#      this won't work when the duplicate is back to back...
#      head -n ${structure_line_numbers[-1]} $1 | tail -$((${structure_line_numbers[-1]} - $duplicate_structure + 1)) >> $final_silent_file
  fi

done

# cat $hbnet_tags_to_grab																														# test
