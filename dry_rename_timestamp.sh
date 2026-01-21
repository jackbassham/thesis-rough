# DRY RUN (print out) of rename all files (.pt, .npz, .png)
# with timestamp YYYYMMDD_HHMMSS to MMDDYYYY_HHMM 

echo "Running loop to rename files with reformatted timestamp"
echo "from 'prefix_YYYYMMDD_HHMMSS.ext' to 'prefix_MMDDYYYY_HHMM.ext'"
echo " " 

# Run loop to rename files with reformatted timestamp
find data/ -type f \( -name "*.pt" -o -name "*.npz"  -o -name "*.png" \) | while read -r f; do
  dir=$(dirname "$f")
  base=$(basename "$f")

# Find the timestamp pattern (matching a digits '_' then 8 digits (from 0-9), then '_' then 4 digits (from 0-9)) after prefix .*
  if [[ $base =~ ^(.*)_([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})[0-9]{2}\.(pt|npz|png)$ ]]; then
    # Assign variables to each BASH_REMATCH in '()'
    prefix="${BASH_REMATCH[1]}"
    year="${BASH_REMATCH[2]}"
    month="${BASH_REMATCH[3]}"
    day="${BASH_REMATCH[4]}"
    hour="${BASH_REMATCH[5]}"
    minute="${BASH_REMATCH[6]}"
    ext="${BASH_REMATCH[7]}"

    # Reorder variables for new timestamp
    new="${prefix}_${month}${day}${year}_${hour}${minute}.${ext}"

    # Print command (DRY RUN!)
    echo mv "$f" "$dir/$new"
  fi
done

echo "Running loop to rename subdirectories with reformatted timestamp"
echo "from 'YYYYMMDD_HHMMSS' to 'MMDDYYYY_HHMM"
echo " "

# Run loop to rename subdirectories with reformatted timestamp
find data/ -type d | sort -r | while read -r d; do
  base=$(basename "$d")
  parent=$(dirname "$d")

  # Find the timestamp pattern
  if [[ $base =~ ^([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})[0-9]*$ ]]; then
    year="${BASH_REMATCH[1]}"
    month="${BASH_REMATCH[2]}"
    day="${BASH_REMATCH[3]}"
    hour="${BASH_REMATCH[4]}"
    minute="${BASH_REMATCH[5]}"

    # Reorder variables for new timestamp
    new="${month}${day}${year}_${hour}${minute}"

    # Print command (DRY RUN!)
    echo mv "$d" "$parent/$new" 
  fi
done