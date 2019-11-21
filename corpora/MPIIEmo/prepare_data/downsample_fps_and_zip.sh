#!/usr/bin/bash
WRITE_DIR="../10fps_views"
READ_DIR="../views"
if [ -d "$WRITE_DIR" ]; then
  echo "remove"
  rm -rf $WRITE_DIR
fi
mkdir $WRITE_DIR

for view_folder in `ls ../views`
 do echo $view_folder
 mkdir $WRITE_DIR/$view_folder
 for file in `ls $READ_DIR/$view_folder`
  do
  ffmpeg -i $READ_DIR/$view_folder/$file -filter:v fps=fps=10 $WRITE_DIR/$view_folder/$file
  done
 zip -r $WRITE_DIR/"$view_folder".zip $WRITE_DIR/$view_folder
 done
