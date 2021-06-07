#! /bin/bash
# from a silent file provided as argument 1 ($1), grab the top X structures integer from argument 2 ($2), using the shape_complementarity and energy_density overlap, and finally the structures which have the highest residue count from this overlap

# create the file which contains the hbnet scores
hbnet_scores=$(dirname -- "$1")/hbnet.sc  # $(basename -- "$1")}
grep "^SCORE" $1 > $hbnet_scores


# find the column number from the scores file
decoy=$(awk -v field=description 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
hbnet_sc=$(awk -v field=sc_hbnet_core 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
khbnet_density=$(awk -v field=ie_density_hbnet_core 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
khbnet_res_count=$(awk -v field=residue_count_hbnet_core 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)
# hbnet_density=$(awk -v field=score 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)								# test
# hbnet_res_count=$(awk -v field=fa_rep 'NR==1{for(i=1 ; i<=NF ; i++){if($i==field){print i;exit}}}' $hbnet_scores)								# test


# sort -k"$khbnet_sc"n -k"$khbnet_density"n $hbnet_scores	# nr sorts each key (k) numerically (n) in descending (r) order->Higher at the top	# test
# sort the scores by the respective column numbers identified, then extract the decoy name
sc_array=($(tail +2 $hbnet_scores | sort -k"$hbnet_sc"nr | head -n $(($2 * 3)) | awk -v field=$decoy '{print $field}'))
density_array=($(tail +2 $hbnet_scores | sort -k"$hbnet_density"n | head -n $(($2 * 3)) | awk -v field=$decoy '{print $field}'))
residue_count_array=($(tail +2 $hbnet_scores | sort -k"$hbnet_res_count"nr | awk -v field=$decoy '{print $field}'))


# sort the elements in both density and sc (shape complementarity) array by the sc order in "overlap"
regex_string=" ${density_array[*]} "            # add framing blanks
for item in ${sc_array[@]}; do
  if [[ $regex_string =~ " $item " ]] ; then    # use $item as regexp
    overlap+=($item)
  fi
done
# incase that the overlap is shorter than the requested amount in $2, add all the members from the sc array
for item in ${sc_array[@]}; do
	overlap+=($item) 
done
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
grep "^SEQUENCE" --max-count=1 $1 > $final_silent_file
grep "^SCORE" --max-count=1 $1 >> $final_silent_file
# rm $final_silent_file


# finally, create the decoys of interest adding each decoy until the specified number provided in $2 is reached
for ((i=0 ; i<$2 ; i++)); do
  echo ${final_overlap[$i]} >> $hbnet_tags_to_grab
  structure_line_numbers=($(grep "${final_overlap[$i]}" -n $1 | cut --fields=1 --delimiter=:))
  head -n ${structure_line_numbers[-1]} $1 | tail -$((${structure_line_numbers[-1]} - ${structure_line_numbers[0]} + 1)) >> $final_silent_file
done

# cat $hbnet_tags_to_grab																														# test
