import sys
import os
import pydrive
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
gauth.SaveCredentialsFile("mycreds.txt")

RAW_BODY_FEATS_DRIVE_ID = constants["RAW_BODY_FEATS_DRIVE_ID"]
download = drive.CreateFile({'id': RAW_BODY_FEATS_DRIVE_ID})
download.GetContentFile("body_feats/raw/all")
