import requests

# Try to acess URL without custom session, and just using ~/.netrc
# NOTE this worked!

URL = 'https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0116_icemotion_vectors_v4/south/daily/icemotion_daily_sh_25km_19890101_19891231_v4.1.nc'

session = requests.Session()
session.trust_env = True   # usually default

r = session.get(URL)

print(r.status_code)