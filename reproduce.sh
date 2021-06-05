#!/bin/bash
set -e

# Download the data.
wget https://surfdrive.surf.nl/files/index.php/s/rzlfzxPMTFEJJ4d/download
mv download data.zip
unzip data.zip
rm data.zip

for LP in "en-de" "de-en" "en-ne" "ne-en" "en-si" "si-en"; do
  DATA_DIR_I="./data/nols_ID/${LP}/"
  EXP_DIR_I="./reproduce/nols_ID/${LP}/"

  echo "Experiment:" $EXP_DIR_I
  mkdir -p $EXP_DIR_I

  python infer-length-model.py "${DATA_DIR_I}" "${EXP_DIR_I}" --svi_steps 50000 --device "cuda:1" | tee "${EXP_DIR_I}/log"
  python check-model.py "${DATA_DIR_I}" "${EXP_DIR_I}/param-store" --device "cuda:1" | tee "${EXP_DIR_I}/checks.txt"
done
