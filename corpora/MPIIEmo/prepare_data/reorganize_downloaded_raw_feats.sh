ALL_RAW_FEATS_DIR="body_feats/raw/all"
RAW_FEATS_DIR="body_feats/raw"

for view_slice in `ls $ALL_RAW_FEATS_DIR`
do
  echo $view_slice
  view=`echo $view_slice | cut -d'-' -f1`
  if [ ! -d "$RAW_FEATS_DIR/$view" ]; then
    mkdir $RAW_FEATS_DIR/$view
  fi
  unzipped_folder=`echo $view_slice | sed 's/.zip//'`
  if [ -d "$unzipped_folder" ]; then
    echo REMOVE
    rm -rf $unzipped_folder
  fi

  unzip -d . $ALL_RAW_FEATS_DIR/$view_slice
  cp -rf "$unzipped_folder"/* $RAW_FEATS_DIR/$view
  rm -rf $unzipped_folder
done
