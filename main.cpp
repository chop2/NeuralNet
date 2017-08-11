#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "neuralnetwork.hpp"
#include "classify/include/plot.hpp"
#include <thread>
using namespace wcv;
void test_network(){

	typedef float _Type;

	_Type data[] = { 11, 12,
		111, 112,
		21, 22,
		211, 212,
		51, 32,
		71, 42,
		441, 412,
		311, 312,
		41, 62,
		81, 52,
		10, 10,
		15, 15,
		12, 14,
		500, 500,
		502, 502,
		504, 504,
		510, 510,
		508, 508
	};
	_Type resp[] = { 1, 0,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1
	};
	//_Type resp[] = { 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1 };

	vector<_Type> vecData(&data[0], &data[sizeof(data) / sizeof(_Type)]);
	vector<_Type> vecResp(&resp[0], &resp[sizeof(resp) / sizeof(_Type)]);

	Matrix<_Type> trainData = Matrix<_Type>(18, 2, vecData);
	Matrix<_Type> response = Matrix<_Type>(18, 2, vecResp);

	wcv::normalize(trainData, trainData, 0, 512);
	//wcv::zero_mean(trainData, trainData);

	Net<_Type, Activation<_Type> > network;
	network << Layer<_Type, Activation<_Type> >(2, 4)
		<< Layer<_Type, Activation<_Type> >(4, 4)
		<< Layer<_Type, Activation<_Type> >(4, 2);
	Net<_Type, Activation<_Type> >::Param param(100000, 0.01, 0.001, 0.0001, 3, true);

	vector<float> vecX;
	vector<float> vecY;
	int epoch = 10;
	namedWindow("err");
	Plot plot;
	plot.vecLegend.push_back("J(0)");
	plot.bdrawText = false;
	auto one_round_callback = [&](int iter, float err){
		if (iter % epoch == 0){
			float x = iter / epoch;
			vecX.push_back(x);
			vecY.push_back(err);
			std::cout << "iter" << iter << ": " << err << endl;
			plot.plot<float>(vecX, vecY, Scalar(255, 0, 0), '+');
			imshow("err", plot.figure0);
			waitKey(30 * 1);
		}
		plot.clear();
	};

	for (size_t i = 0; i < network.getNetSize(); i++) {
		Layer<_Type, Activation<_Type>>& layer = network.getLayer(i);
		cout << "Layer(" << i << "):\n" << layer.getWeights() << endl << endl;
	}

	network.train(trainData, response, param, one_round_callback);

	cout << network.getLayers() << endl;
	for (size_t i = 0; i < network.getNetSize(); i++) {
		Layer<_Type, Activation<_Type>>& layer = network.getLayer(i);
		cout << "Layer(" << i << "):\n" << layer.getWeights() << endl << endl;
	}

	// Data for visual representation  
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);
	Vec3b green(0, 255, 0), blue(255, 0, 0);
	// Show the decision regions
	for (int i = 0; i < image.rows; ++i)
	{
		Vec3b* p = (Vec3b*)image.ptr<Vec3b>(i);
		for (int j = 0; j < image.cols; ++j)
		{
			Matrix<_Type> m = Matrix<_Type>(1, 2);
			m.at(0, 0) = i;
			m.at(0, 1) = j;
			wcv::normalize(m, m, 0, 512);

			Matrix<_Type> rsp = network.forward(m);
			//
			if (rsp.at(0, 0) > rsp.at(0, 1))	{
				p[j] = green;
			}
			else{
				p[j] = blue;
			}
		}
	}

	// Show the training data  
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(111, 112), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(211, 212), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(441, 412), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(311, 312), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(11, 12), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(21, 22), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(51, 32), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(71, 42), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(41, 62), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(81, 52), 5, Scalar(255, 255, 255), thickness, lineType);

	imwrite("result.png", image);        // save the image   

	imshow("BP Simple Example", image); // show it to the user  
	waitKey(0);
}

#include <emmintrin.h>
void test_see()
{
	Mat src = imread("d:/test.jpg", 0);
	Mat img = src.clone();
	int val = 100;
	for (size_t i = 0; i < src.rows; i++)
	{
		uchar* p = (uchar*)src.ptr() + src.step * i;
		for (size_t j = 0; j < src.cols - 16; j += 16)
		{
			__m128i v0, v1, v;
			v = _mm_set_epi16(val, val, val, val, val, val, val, val);
			v0 = _mm_loadu_si128((const __m128i*)(p + j));
			v1 = _mm_loadu_si128((const __m128i*)(p + j + 8));
			v0 = _mm_add_epi8(v0, v);
			v1 = _mm_add_epi8(v1, v);
			_mm_storeu_si128((__m128i*)(p + j), v0);
			_mm_storeu_si128((__m128i*)(p + j + 8), v1);
		}
	}

	Mat ii = src.clone();
}

#include "classify\include\classify.hpp"
void test_digit()
{
	std::string samplePth = "D:\\Char\\print_char\\Calibri\\digit";
	Size normSize(8, 16);
	Ptr<FeatureExtractor<dtype> > pfeat = FeatureFactory::createPixelFeatureExtractor<dtype>(false);
	cv::Ptr<TrainDataMaker<dtype> > dataMaker = DataMakerFactory<dtype>::createDataMaker(
		pfeat, normSize, 0.3, true);
	dataMaker->extractSamples(samplePth);
	cv::Mat cvTrainData = dataMaker->getTrainData();
	cv::Mat cvTrainRsp = dataMaker->getTrainResponses();

	Matrix<dtype> trainData = Matrix<dtype>(cvTrainData);
	Matrix<dtype> trainRsp = Matrix<dtype>(cvTrainRsp);
	int nInputs = normSize.area();
	int nOutputs = dataMaker->getNumClasses();
	int nHiddent = 4;
	Net<dtype, Activation<dtype> > network;
	network << Layer<dtype, Activation<dtype> >(nInputs, nHiddent)
		<< Layer<dtype, Activation<dtype> >(nHiddent, nHiddent)
		<< Layer<dtype, Activation<dtype> >(nHiddent, nOutputs);
	Net<dtype, Activation<dtype> >::Param param(100, 0.01, 0.001, 0.0001, 3, true);

	vector<float> vecX;
	vector<float> vecY;
	int epoch = 10;
	namedWindow("err");
	Plot plot;
	plot.vecLegend.push_back("J(0)");
	plot.bdrawText = false;
	auto one_round_callback = [&](int iter, float err){
		if (iter % epoch == 0 /*&& iter != 0*/){
			float x = iter / epoch;
			vecX.push_back(x);
			vecY.push_back(err);
			std::cout << "iter" << iter << ": " << err << endl;
			plot.plot<float>(vecX, vecY, Scalar(255, 0, 0), '+');
			imshow("err", plot.figure0);
			waitKey(30 * 1);
		}
		plot.clear();
	};

	//for (size_t i = 0; i < network.getNetSize(); i++) {
	//	Layer<dtype, Activation<dtype>>& layer = network.getLayer(i);
	//	cout << "Layer(" << i << "):\n" << layer.getWeights() << endl << endl;
	//}

	network.train(trainData, trainRsp, param, one_round_callback);

	//cout << network.getLayers() << endl;
	//for (size_t i = 0; i < network.getNetSize(); i++) {
	//	Layer<dtype, Activation<dtype>>& layer = network.getLayer(i);
	//	cout << "Layer(" << i << "):\n" << layer.getWeights() << endl << endl;
	//}

	cv::Mat cvTestData = dataMaker->getTestData();
	cv::Mat cvTestRsp = dataMaker->getTestResponses();

	Matrix<dtype> testData = Matrix<dtype>(cvTestData);
	Matrix<dtype> testRsp = Matrix<dtype>(cvTestRsp);
	Matrix<int> err_grid;
	double err = network.calcError(testData, testRsp, &err_grid);
	err_grid = network.formatErrGrid(err_grid);
	std::cout << "error = " << err << endl;
	std::cout << "error grid: \n" << err_grid << endl;
	network.save("ann.model");
	network.load("ann.model");
	system("pause");
}

int main()
{
	//test_see();
	test_digit();
	test_network();
}