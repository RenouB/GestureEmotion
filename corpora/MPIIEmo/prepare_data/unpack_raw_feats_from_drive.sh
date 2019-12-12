# download project folder from google drive into folder
# drive_download in project parent directory
#
# https://drive.google.com/open?id=12tf83gQiewTNKgiyfnC4wUlWXZ6zp6rL

# run this script from prepare_data directory
# it will unzip raw annotations and place them in correct location

cd ../../../drive_download/
if [ ! -d "./raw_features" ]; then
  echo "raw_features dir not detected; unzipping downloaded drive files"
  unzip ./\*.zip
  mkdir zips
  mv *.zip zips
fi

cd raw_features

TEN_FPS_VIEWS_DIR=../../corpora/MPIIEmo/views/10fps_views
RAW_BODY_FEATS_DIR=../../corpora/MPIIEmo/features/body_features/raw
VERIFICATION_IMAGES_DIR=../../corpora/MPIIEmo/prepare_data/verification_images

# unzip raw feats into correct folder
if [ -d "$RAW_BODY_FEATS_DIR" ]; then
  echo "removing existing raw body feats dir"
  rm -rf $RAW_BODY_FEATS_DIR
fi
mkdir -p $RAW_BODY_FEATS_DIR

for file in `ls *body-features.zip`
do
  mkdir tmp
  view=`echo $file | cut -d'/' -f2 | cut -d'-' -f1`
  tmp_subfolder=`echo $file | cut -d'/' -f2 | sed 's/\.zip//'`
  mkdir -p $RAW_BODY_FEATS_DIR/$view
  echo $file
  echo  $RAW_BODY_FEATS_DIR/$view
  unzip -d tmp $file
  mv tmp/$tmp_subfolder/* $RAW_BODY_FEATS_DIR/$view
  rm -rf tmp
done

for view in `ls $RAW_BODY_FEATS_DIR`
do
  echo $view
  ls $view | wc
done

# stuff below is deprecated
#
# # # unzip 10fps_views files into correct folder
# if [ -d "$TEN_FPS_VIEWS_DIR" ]; then
#   rm -rf $TEN_FPS_VIEWS_DIR
# fi
# mkdir -p $TEN_FPS_VIEWS_DIR
#
# for file in `ls *10fps.zip`
# do
#   mkdir tmp
#   view=`echo $file | cut -d'_' -f1`
#   mkdir -p $TEN_FPS_VIEWS_DIR/$view
#   unzip -d tmp $file
#   mv tmp/10fps_views/$view/* $TEN_FPS_VIEWS_DIR/$view/
#   rm -rf tmp
# done

# this whole block was here because I wanted to download verification images
# but this idea was dropped

# # unzip verification images into correct folder
# if [ -d "$VERIFICATION_IMAGES_DIR" ]; then
#   rm -rf $VERIFICATION_IMAGES_DIR
# fi
# mkdir -p $VERIFICATION_IMAGES_DIR
#
# for file in `ls raw_features/*images.zip`
# do
#   mkdir tmp
#   tmp_subfolder=`echo $file | cut -d'/' -f2 | sed 's/\.zip//'`
#   view=`echo $file | cut -d'/' -f2 | cut -d'-' -f1`
#   mkdir $VERIFICATION_IMAGES_DIR/$view
#   unzip -d tmp $file
#   mv tmp/$tmp_subfolder/* $VERIFICATION_IMAGES_DIR/$view
#   rmdir -rf tmp
# done
#
# cd ../../
# echo "Now randomly selecting verification images"
# python corpora/MPIIEmo/prepare_data/randomly_select_verification_images.py
