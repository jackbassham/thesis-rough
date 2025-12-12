from datetime import datetime

# Generate time stamp with format #ddMMMYY_HHMM
timestamp = datetime.now().strftime("%d%b%Y_%H%M")

# Print timestamp for bash script
print(timestamp)