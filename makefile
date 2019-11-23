all: Train_cnn_multi Test_cnn Test_cnn_transE
Train_cnn_multi: Train_cnn_multi.cpp
	g++ Train_cnn_multi.cpp -o Train_cnn_multi -O2 -lpthread
Test_cnn: Test_cnn.cpp
	g++ Test_cnn.cpp -o Test_cnn -O2 -lpthread
Test_cnn_transE: Test_cnn_transE.cpp
	g++ Test_cnn_transE.cpp -o Test_cnn_transE -O2 -lpthread
clean:
	rm -rf Train_cnn_multi Test_cnn Test_cnn_transE
