#!/bin/bash
dir="ppm/"
ftype=".ppm"
for i in "$dir"*"$ftype"; do
  name=${i#"$dir"}
  name=${name%"$ftype"}
  echo "transforming $name"
  pnmtojpeg --quality=95 $i > jpg/"${name}.jpg"
done
