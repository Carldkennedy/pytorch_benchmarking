#!/bin/bash
rm -rf calc/
rm stats.out
mkdir calc

num_runs=$(($1))

for file in *.out; do
    # Collect lines containing "Epoch" and "time"
    grep -E "Epoch.*time" "$file" > "calc/Times_$file"
    
    # Check if the 5th to last line contains "Finished"
    if tail -n 5 "$file" | grep -q "Finished"; then
      
        # Extract 7th column
        cut -d' ' -f7 "calc/Times_$file" > "calc/times_$file"

        # Calculate average of consecutive 20 lines (epochs)
        awk '{ total += $1 } NR % 20 == 0 { print total/20; total = 0 }' "calc/times_$file" > "calc/avg_$file"

        # Calculate standard deviation of average values
        std_dev=$(awk '{ total += $1; total2 += ($1)^2; n++ } END { mean = total / n; print sqrt(total2 / n - (mean^2)) }' "calc/avg_$file")

        avg_all=$(awk '{ total += $1 } END { print total/NR }' "calc/times_$file")
        # Print filename and average and standard deviation on the same line
        echo "$file $avg_all $std_dev" >> stats.out

    fi
done
