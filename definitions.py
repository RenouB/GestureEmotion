import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

constants = \
{
"PROJECT_DIR" : PROJECT_DIR,
"MPIIEMO_DATA_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/data"),
"MPIIEMO_ANNOS_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/annos_website"),
"TEN_FPS_VIEWS_DIR" : os.path.join(PROJECT_DIR, "corpora/MPIIEmo/10fps_views")
}
