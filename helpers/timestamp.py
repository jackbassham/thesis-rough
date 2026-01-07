from datetime import datetime

# Generate time stamp with format #ddMMMYY_HHMM
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Print timestamp for bash script
print(timestamp)