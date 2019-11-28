import os
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



TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]

paths = [os.path.join(TEN_FPS_VIEWS_DIR, "view{}.zip".format(i)) for i in range(1,9)]
filenames = ["view{}_10fps.zip".format(i) for i in range(1,9)]

with open("MPIIEMO.id", 'r') as f:
	parent_id = f.read()[:-1]

def upload_file_to_specific_folder(self, folder_id, file_path, file_name):
    file_metadata = {'title': file_name, "parents": [{"id": folder_id, "kind": "drive#childList"}]}
    folder = drive.CreateFile(file_metadata)
    folder.SetContentFile(file_path)
    folder.Upload()

for i in range(len(paths)):
    print("Writing view {}".format(i))
    upload_file_to_specific_folder(drive, parent_id, paths[i], filenames[i])
