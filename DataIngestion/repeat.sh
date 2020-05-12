#!/bin/bash

#for i in {2..16}
#do
#  echo "Processing sub_dir $i..."
#  python ./ComputeModelDetection_Linear.py --gw_id S190814bv --healpix_file LALInference.v1.fits.gz,0 --num_cpu 6 --sub_dir $i
#  echo "sub_dir $i complete!"
#done


#echo "Processing on-axis models..."
#for i in {3..10}
#do
#  echo "Processing sub_dir $i..."
#  python ./ComputeModelDetection_GRB.py --gw_id S190814bv --healpix_file LALInference.v1.fits.gz,0 --num_cpu 6 --sub_dir grb_onaxis --batch_dir $i --merger_time_MJD 58709.882824224536
#  echo "sub_dir $i complete!"
#done
#
#
#echo "Processing off-axis models..."
#for i in {1..10}
#do
#  echo "Processing sub_dir $i..."
#  python ./ComputeModelDetection_GRB.py --gw_id S190814bv --healpix_file LALInference.v1.fits.gz,0 --num_cpu 6 --sub_dir grb_offaxis --batch_dir $i --merger_time_MJD 58709.882824224536
#  echo "sub_dir $i complete!"
#done

#for i in {2..596}
for i in {2..2}
do
  echo "Processing sub_dir $i..."
  python ./ComputeModelDetection_KN.py --model_base_dir /data2/ckilpatrick/KNmodel/tables --num_cpu 18 --sub_dir $i
  echo "sub_dir $i complete!"
done