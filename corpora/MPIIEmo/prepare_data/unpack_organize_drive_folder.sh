

# download project folder from google drive into folder
# drive_download in project parent directory
#
# https://drive.google.com/open?id=1rEFKedGwxqhQl0Z6fSRpaX9H058h6KAZ

# run this script from prepare_data directory

cd ../../../drive_download/
# unzip all the zips downloaded from drive
# unzip ./\*.zip
cd MPIIEmo
TEN_FPS_VIEWS_DIR=../../corpora/MPIIEmo/views/10fps_views
RAW_BODY_FEATS_DIR=../../corpora/MPIIEmo/features/body_features/raw
VERIFICATION_IMAGES_DIR=../../corpora/MPIIEmo/prepare_data/verification_images

# # unzip 10fps_views files into correct folder
# if [ -d "$TEN_FPS_VIEWS_DIR" ]; then
#   rm -rf $TEN_FPS_VIEWS_DIR
#   mkdir $TEN_FPS_VIEWS_DIR
# fi
#
# for file in `ls *10fps.zip`
# do
#   view=`echo $file | cut -d'_' -f1`
#   mkdir $TEN_FPS_VIEWS_DIR/$view
#   unzip -d $TEN_FPS_VIEWS_DIR/$view/ $file
# done

# unzip raw feats into correct folder
if [ -d "$RAW_BODY_FEATS_DIR" ]; then
  echo "removing raw body feats dir"
  rm -rf $RAW_BODY_FEATS_DIR
  mkdir $RAW_BODY_FEATS_DIR
fi
echo "removed raw body feats dir"

for file in `ls raw_features/*body-features.zip`
do
  view=`echo $file | cut -d'_' -f1`
  mkdir $RAW_BODY_FEATS_DIR/$view
  unzip -d $RAW_BODY_FEATS_DIR/$view/ $file
done
#
# # unzip verification images into correct folder
# if [ -d "$VERIFICATION_IMAGES_DIR" ]; then
#   rm -rf $VERIFICATION_IMAGES_DIR
#   mkdir $VERIFICATION_IMAGES_DIR
# fi
#
# for file in `ls raw_features/*images.zip`
# do
#   view=`echo $file | cut -d'_' -f1`
#   mkdir $VERIFICATION_IMAGES_DIR/$view
#   unzip -d $VERIFICATION_IMAGES_DIR/$view/ raw_features/$file
# done
