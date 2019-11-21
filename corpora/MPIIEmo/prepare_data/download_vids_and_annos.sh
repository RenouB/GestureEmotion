mkdir ../views
WRITE_DIR=../views

wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view1.zip -P $WRITE_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view2.zip -P $WRITE_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view3.zip -P $WRITE_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view4.zip -P  $WRITE_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view5.zip -P $WRITE_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view6.zip -P $WRITE_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view7.zip -P $WRITE_DIR
wget http://www.datasets.d2.mpi-inf.mpg.de/MPIIEmo/videos/view8.zip -P $WRITE_DIR

mkdir ../zips
unzip $WRITE_DIR/\*.zip ../zips

wget http://transfer.d2.mpi-inf.mpg.de/MPIIEmo/annotations.zip -P ..
rm -rf ../annos_website
unzip ../annotations.zip -d ..
rm ../annotations.zip
