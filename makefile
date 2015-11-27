all: Train_cnn_multi Test_cnn
Train_cnn_multi: Train_cnn_multi.cpp
	g++ Train_cnn_multi.cpp -o Train_cnn_multi -O2 -lpthread
Test_cnn: Test_cnn.cpp
	g++ Test_cnn.cpp -o Test_cnn -O2 -lpthread
clean:
	rm -rf Train_cnn_multi Test_cnn