cd FaceBoxes
sh ./make.sh
cd ..

cd Sim3DR
sh ./build_sim3dr.sh
cd ..

cd utils/asset
gcc -shared -Wall -O3 render.c -o render.so -fPIC
cd ../..
