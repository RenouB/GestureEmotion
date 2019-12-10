# download project folder from google drive into folder
# drive_download in project parent directory
#
# https://drive.google.com/open?id=1rEFKedGwxqhQl0Z6fSRpaX9H058h6KAZ

# run this script from prepare_data directory
# it will unzip 10fps views, raw body annotations and verification images
# and put them into correct folders

cd ../../../drive_download/
cd MPIIEmo

TEN_FPS_VIEWS_DIR=../../corpora/MPIIEmo/views/10fps_views
RAW_BODY_FEATS_DIR=../../corpora/MPIIEmo/features/body_features/raw
VERIFICATION_IMAGES_DIR=../../corpora/MPIIEmo/prepare_data/verification_images

# # unzip 10fps_views files into correct folder
if [ -d "$TEN_FPS_VIEWS_DIR" ]; then
  rm -rf $TEN_FPS_VIEWS_DIR
fi
mkdir $TEN_FPS_VIEWS_DIR

for file in `ls *10fps.zip`
do
  mkdir tmp
  view=`echo $file | cut -d'_' -f1`
  mkdir $TEN_FPS_VIEWS_DIR/$view
  unzip -d tmp $file
  mv 10fps_views/$view/* $TEN_FPS_VIEWS_DIR/$view/
  rm -rf tmp
done

# unzip raw feats into correct folder
if [ -d "$RAW_BODY_FEATS_DIR" ]; then
  echo "removing raw body feats dir"
  rm -rf $RAW_BODY_FEATS_DIR
fi
mkdir $RAW_BODY_FEATS_DIR


for file in `ls raw_features/*body-features.zip`
do
  mkdir tmp
  view=`echo $file | cut -d'/' -f2 | cut -d'-' -f1`
  tmp_subfolder=`echo $file | cut -d'/' -f2 | sed 's/\.zip//'`
  mkdir $RAW_BODY_FEATS_DIR/$view
  echo $file
  echo  $RAW_BODY_FEATS_DIR/$view
  unzip -d tmp $file
  mv tmp/$tmp_subfolder/* $RAW_BODY_FEATS_DIR/$view
  rm -rf tmp
done

# unzip verification images into correct folder
if [ -d "$VERIFICATION_IMAGES_DIR" ]; then
  rm -rf $VERIFICATION_IMAGES_DIR
fi
mkdir $VERIFICATION_IMAGES_DIR

for file in `ls raw_features/*images.zip`
do
  mkdir tmp
  tmp_subfolder=`echo $file | cut -d'/' -f2 | sed 's/\.zip//'`
  view=`echo $file | cut -d'/' -f2 | cut -d'-' -f1`
  mkdir $VERIFICATION_IMAGES_DIR/$view
  unzip -d tmp $file
  mv tmp/$tmp_subfolder/* $VERIFICATION_IMAGES_DIR/$view
  rmdir -rf tmp
done
