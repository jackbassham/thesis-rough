from datetime import datetime

# Generate time stamp with format #MMDDYY_HHMM
timestamp = datetime.now().strftime("%m%d$Y_%H%M")

# Print timestamp for bash script
print(timestamp)