import os

# create virual environment and install dependencies
# pip install -r requirements.txt

# download the raw body features from
# https://drive.google.com/open?id=12tf83gQiewTNKgiyfnC4wUlWXZ6zp6rL
# and place into drive_download

os.system('cd corpora/MPIIEmo/prepare_data && sh unpack_raw_feats_from_drive.sh && python process_raw_body_feats.py")
