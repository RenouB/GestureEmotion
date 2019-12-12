# run from prepare_data directory
# will download and unzip original videos and annotations

 if [ -d ../views/original_views]; then
   rm -rf ../views/original_views
   mkdir -p ../views/original_views
 fi

VIEWS_DIR=../views/original_views

cd views
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view1.zip -P $VIEWS_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view2.zip -P $VIEWS_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view3.zip -P $VIEWS_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view4.zip -P  $VIEWS_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view5.zip -P $VIEWS_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view6.zip -P $VIEWS_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view7.zip -P $VIEWS_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view8.zip -P $VIEWS_DIR


unzip $VIEWS_DIR/\*.zip

wget http://transfer.d2.mpi-inf.mpg.de/MPIIEmo/annotations.zip -P ..
rm -rf ../annos_website
unzip ../annotations.zip -d ..
rm ../annotations.zip
