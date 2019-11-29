import os
import json
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(PROJECT_DIR,"corpora/MPIIEmo/prepare_data/body25_keypoint_mapping.json")) as f:
    body_mapping = json.load(f)["mapping"]

constants = \
{
"PROJECT_DIR" : PROJECT_DIR,
"MPIIEMO_DATA_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/data"),
"MPIIEMO_ANNOS_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/annos_website"),
"TEN_FPS_VIEWS_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/10fps_views"),
"RAW_BODY_FEATS_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/prepare_data/body_feats/raw"),
"PROCESSED_BODY_FEATS_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/prepare_data/body_feats/processed"),
"MPIIEMO_ID_PATH" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/prepare_data/MPIIEMO.id"),
"KEYPOINTS_FOR_SCALING" : (9, 15),
"RAW_BODY_FEATS_DRIVE_ID" : "12tf83gQiewTNKgiyfnC4wUlWXZ6zp6rL",
"PAFS" : [(17, 15), (15, 0), (0, 16), (16, 18), (0,1), (1, 2), (2, 3), (3, 4),
          (1,5), (5,6), (6,7), (1,8), (8,9), (9, 10), (10, 11), (11, 24), (11, 22),
          (22, 23), (8,12), (12,13), (13,14), (14,21), (14,19), (19, 20)],
"BODY_KEYPOINT_MAPPING" : body_mapping,
"STABLE_BODY_PART_INDICES" : [0, 1, 2, 5, 8, 9, 12, 15, 10, 17, 18]
}
