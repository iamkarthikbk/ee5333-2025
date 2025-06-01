#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p output

# Create a log file for statistics
log_file="output/routing_stats.log"
echo "Routing Statistics - $(date)" > "$log_file"
echo "----------------------------------------" >> "$log_file"

# Loop through all DEF files in the def/ directory
for def_file in def/*.def; do
    # Extract the base filename without extension
    base_name=$(basename "$def_file" .def)
    
    # Skip if the filename ends with "_routed"
    if [[ "$base_name" == *"_routed"* ]]; then
        echo "Skipping already routed file: $def_file"
        continue
    fi
    
    # Define the corresponding guide file
    guide_file="gr/${base_name}.guide"
    
    # Define output files
    output_file="output/${base_name}.def"
    checked_file="output/${base_name}_checked.def"
    temp_output="output/${base_name}_checker_output.tmp"
    
    # Check if guide file exists
    if [ ! -f "$guide_file" ]; then
        echo "Warning: Guide file $guide_file not found for $def_file. Skipping."
        continue
    fi
    
    echo "Processing $base_name..."
    echo "Running router for $def_file with guide $guide_file..."
    
    # Time the router execution
    start_time=$(date +%s)
    python router.py -l lef/sky130.lef -d "$def_file" -g "$guide_file" -o "$output_file" > /dev/null
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    
    # Run the checker and capture output
    echo "Running checker for $def_file..."
    python checker.py -i "$def_file" -o "$output_file" -l lef/sky130.lef > "$temp_output" 2>&1
    
    # Extract the last 3 lines from checker output
    echo "" >> "$log_file"
    echo "$base_name Statistics:" >> "$log_file"
    echo "Router execution time: $execution_time seconds" >> "$log_file"
    tail -n 3 "$temp_output" >> "$log_file"
    echo "----------------------------------------" >> "$log_file"
    
    # Clean up temporary file
    rm -f "$temp_output"
    
    echo "Completed processing $base_name (took $execution_time seconds)"
    echo "----------------------------------------"
done

echo "All files processed!"
echo "Statistics saved to $log_file"

# Display the statistics
echo ""
echo "Routing Statistics Summary:"
cat "$log_file"
