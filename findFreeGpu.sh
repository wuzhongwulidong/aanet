#!/bin/bash
echo '[Info] Please ensure that: conda env is Activated.'
neededMem=11000
neededGPUs=1
targetShell='./scripts/Diag_mini_modify_AANet_train.sh'

echo "neededMem=$neededMem MB"
echo "neededGPUs=$neededGPUs"
echo "targetShell=$targetShell"
echo ""

# Empty array to store free GPU ids.
freeGPUs=()

while [ ${#freeGPUs[*]} -lt $neededGPUs ]
do
   freeGPUs=()
   usedMem=$(nvidia-smi | grep "Default" | awk '{print $9}')

   id=0
   for elem in $usedMem
   do
      #echo $elem
	  #Get the size of used memory, MB.
      usedSize=$(echo $elem | tr -cd "[0-9]")
      # Available Memory on this gpu. unit MB.
      freeSize=$((12196-usedSize))
      echo $freeSize
   
      # if
      if [ "$freeSize" -gt "$neededMem" ];then
         echo "GPU id=$id , available memory is $freeSize MB > $neededMem MB"
	     
         freeGPUs[${#freeGPUs[@]}]=$id   
	     
         #echo "elements freeGPUs: ${freeGPUs[@]}"   
      fi
    
      let ++id

      if [ ${#freeGPUs[*]} -eq $neededGPUs ];then
         break 
      fi

   done
   
   if [ ${#freeGPUs[@]} -lt $neededGPUs ]
   then
   echo "Only ${#freeGPUs[@]} GPUs are free, but I need $neededGPUs GPUs."
   # Sleep a while
   echo '===================Sleeping 10 seconds============================'
   sleep 10s
   echo ''
   fi   

done

freeGPUs_str=''
for i in ${freeGPUs[@]};
do 
   freeGPUs_str=$freeGPUs_str$i','
done
# Delete the last comma ','
freeGPUs_str=${freeGPUs_str%?}

#echo "Enough free GPUs detected, GPU id: $freeGPUs_str"

export CUDA_VISIBLE_DEVICES=$freeGPUs_str
echo "Enough free GPUs detected, going to using GPU id: $CUDA_VISIBLE_DEVICES"

dos2unix $targetShell

source $targetShell



