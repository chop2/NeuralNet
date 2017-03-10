#include <vector>
#include <string>
#include <assert.h>
#include <random>
#include <memory>
using namespace std;

#define HAVE_OPENCV
#ifdef HAVE_OPENCV
#include <opencv2/core/core.hpp>
#endif

template<typename dtype> class Tensor;
namespace wcv{
	template<typename dtype>
	float randn(float mean = 0.f,float sigma = 1.f){
		std::default_random_engine re(time(NULL));
		std::normal_distribution<float> d(mean, sigma);
		return d(re);
	}
	template<typename dtype>
	void randn(Tensor<dtype>& _Out,float mean = 0.f,float sigma = 1.f){
		assert(_Out.shape().size() > 3);
		int N = _Out.shape()[0];
		int C = _Out.shape()[1];
		std::default_random_engine re(time(NULL));
		std::normal_distribution<float> d(mean, sigma);
		for (size_t n = 0; n < N; ++n) {
			for (size_t c = 0; c < C; ++c) {
				dtype* pImg = _Out.offset(n, c);
				size_t spatial_dim = _Out.count(2);
				for (size_t k = 0; k < spatial_dim; ++h) {
					pImg[k] = static_cast<dtype>(d(re));
				}
			}
		}
	}
	template<typename dtype>
	void rand(Tensor<dtype>& _Out, float _Min = 0.f, float _Max = 1.f){
		assert(_Out.shape().size() > 3);
		int N = _Out.shape()[0];
		int C = _Out.shape()[1];
		std::default_random_engine re(time(NULL));
		std::uniform_real_distribution<float> d(_Min, _Max);
		for (size_t n = 0; n < N; ++n) {
			for (size_t c = 0; c < C; ++c) {
				dtype* pImg = _Out.offset(n, c);
				size_t spatial_dim = _Out.count(2);
				for (size_t k = 0; k < spatial_dim; ++k) {
					pImg[k] = static_cast<dtype>(d(re));
				}
			}
		}
	}
	template<typename dtype>
	void fill(Tensor<dtype>& _Out, float _Val = 0.f){
		size_t _Size = _Out.count();
		dtype* pdata = _Out.mutable_data();
		for (size_t i = 0; i < _Size; ++i) {
			pdata[i] = static_cast<dtype>( _Val );
		}
	}
}



#define TSR_MIN_DIM 4
#define TSR_CHANNEL_AXIS 1
template<typename dtype=float>
class Tensor
{
public:
	Tensor() :_shape(), _vdata(){};
	Tensor(const vector<int> dim,dtype* data=nullptr){
		reshape(dim);
		if (data != nullptr){
			for (size_t i = 0; i < count(); ++i) {
				_vdata[i] = data[i];
			}
		}
	}
#ifdef HAVE_OPENCV
	Tensor(const vector<cv::Mat>& mvs){
		_shape.resize(4);
		_shape[0] = mvs.size();
		_shape[1] = mvs[0].channels();
		_shape[2] = mvs[0].rows;
		_shape[3] = mvs[0].cols;
		for (size_t n = 0; n < _shape[0]; ++n) {
			cv::Mat& m = mvs[n];
			vector<cv::Mat> vm;
			if (m.channels() > 1){
				cv::split(m, vm);
			} else{
				vm.push_back(m);
			}
			for (size_t c = 0; c < _shape[1]; ++c){
				vector<dtype> vd = mat2dtype(vm[c]);
				_vdata.insert(_vdata.end(), vd.begin(), vd.end());
			}
		}
	}
	vector<dtype> mat2dtype(const cv::Mat& mat){
		assert(mat.type() == CV_32FC1);
		std::vector<float> array;
		if (mat.isContinuous()) {
			if (mat.type() == CV_32FC1)
				array.assign((float*)mat.datastart, 
				(float*)mat.dataend);
		} else {
			for (int i = 0; i < mat.rows; ++i) {
				if (mat.type() == CV_32FC1)
					array.insert(array.end(), (float*)mat.ptr<uchar>(i), 
					(float*)mat.ptr<uchar>(i)+mat.cols);
			}
		}
		std::vector<float> arrayd(array.size());
		for (size_t i = 0; i < array.size(); i++) {
			arrayd[i] = static_cast<dtype>(array[i]);
		}
	}
#endif
	~Tensor(){ release(); };
	void reshape(const vector<int>& dim){
		assert(dim.size() >= TENSOR_MIN_DIM);
		_shape = dim;
		_vdata.resize(dim());
	}
	void reshape(const Tensor& rhs){
		_shape = rhs._shape;
		_vdata.resize(rhs._vdata.size());
	}
	const dtype* const_data(){
		return _vdata.data();
	}
	dtype* mutable_data(){
		const_cast<dtype*>(_vdata.data());
	}
	dtype* offset(int n, int c){
		return const_cast<dtype*>(_vdata.data() + n * channels() + c);
	}
	void append(const Tensor& rhs){
		assert(_shape.size() == rhs.size());
		bool bsm = true;
		for (size_t i = TSR_CHANNEL_AXIS; i < _shape.size(); ++i) {
			if (_shape[i] != rhs._shape[i]){
				bsm = false; break;
			}
		}
		assert(bsm == true);
		_shape[0] += rhs._shape[0];
		_vdata.insert(_vdata.end(), rhs._vdata.begin(), rhs._vdata.end());
	}
	dtype& at(int n = 0, int c = 0, int h = 0, int w = 0) const{
		return _vdata[((n*channels() + c)*height() + h)*width() + w];
	}
	vector<int>& shape() const{
		return _shape;
	}
	size_t count(int startAxies = 0){
		assert(!_shape.empty());
		size_t num_spatial_dim = 1;
		for (size_t i = startAxies; i < _shape.size(); i++) {
			num_spatial_dim *= _shape[i];
		}
		return num_spatial_dim;
	}
	int num(){
		assert(!_shape.empty() && _shape.size() > 1);
		return _shape[0];
	}
	int channels(){
		assert(!_shape.empty() && _shape.size() > 2);
		return _shape[1];
	}
	int height(){
		assert(!_shape.empty() && _shape.size() > 3);
		return _shape[2];
	}
	int width(){
		assert(!_shape.empty() && _shape.size() > 4);
		return _shape[3];
	}
	void release(){
		_shape.clear();
		_vdata.clear();
	}
private:
	vector<int> _shape;
	vector<dtype> _vdata;
};

class LayerParam
{
public:
	LayerParam() :_num_output(0){};
	LayerParam(int num_output) :_num_output(num_output){};
	~LayerParam(){};
protected:
	int _num_output;
};

class ConvParam :public LayerParam
{
public:
	ConvParam() :LayerParam(0), _kernel_size(0), _kernel_h(0),
		_kernel_w(0), _stride(0), _pad(0){};
	ConvParam(int num_output, int kernel_size, int stride, int pad) :
		LayerParam::LayerParam(num_output), _kernel_size(kernel_size),
		_kernel_h(kernel_size), _kernel_w(kernel_size), _stride(stride), _pad(pad){};
	ConvParam(int num_output, int kernel_h, int kernel_w, int stride, int pad) :
		LayerParam::LayerParam(num_output), _kernel_size(0),
		_kernel_h(kernel_h), _kernel_w(kernel_w),
		_stride(stride), _pad(pad){};
	~ConvParam(){};
private:
	int _kernel_size;
	int _kernel_h;
	int _kernel_w;
	int _stride;
	int _pad;
};

template<typename dtype>
class Layer
{
public:
	Layer();
	~Layer();
	virtual void Reshape(const vector<Tensor<dtype>* >& bottom, const vector<Tensor<dtype>* >& top) = 0;
	virtual void Forward(const vector<Tensor<dtype>* >& bottom, const vector<Tensor<dtype>* >& top) = 0;
	virtual void Backward(const vector<Tensor<dtype>* >& top, const vector<Tensor<dtype>* >& bottom) = 0;
protected:
	void Init(){};
protected:
	vector<Tensor<dtype> > _diff;
	vector<Tensor<dtype> > _data;
	vector<Tensor<dtype> > _weights;//_weights[0]->kernel weight ,_weights[1]->bias
};

template<typename dtype>
class ConvLayer:public Layer<dtype>
{
public:
	ConvLayer() :_diff(), _data(){};
	ConvLayer(LayerParam& _Param){
		ConvParam* convParam = dynamic_cast<ConvParam*>(&_Param);
		assert(convParam != nullptr);
		_num_output = convParam->_num_output;
		_kernel_size = convParam->_kernel_size;
		_kernel_h = convParam->_kernel_h;
		_kernel_w = convParam->_kernel_w;
		_stride = convParam->_stride;
		_pad = convParam->_pad;
		Init();
	}
	ConvLayer(int num_output, int kernel_size, int stride, int pad) :
		_num_output(num_output), _kernel_size(kernel_size), _kernel_h(kernel_size), 
		_kernel_w(kernel_size), _stride(stride), _pad(pad) {
		Init();
	}
	ConvLayer(int num_output, int kernel_h, int kernel_w,int stride, int pad) :
		_num_output(num_output), _kernel_size(0),
		_kernel_h(kernel_h), _kernel_w(kernel_w),
		_stride(stride), _pad(pad){
		Init();
	}
	~ConvLayer(){};
	virtual void Reshape(const vector<Tensor<dtype>* >& bottom,
		const vector<Tensor<dtype>* >& top) {
		_diff.resize(bottom->size());
		_data.resize(bottom->size());
		vector<int> input_shape = bottom[0]->shape();
		vector<int> output_shape;
		calc_output_shape(input_shape, output_shape);
		for (size_t i = 0; i < top.size(); ++i)	{
			top[i]->reshape(output_shape);
			_diff[i] = Tensor<dtype>(output_shape);
			_data[i] = Tensor<dtype>(output_shape);
		}
	}
	virtual void Forward(const vector<Tensor<dtype>* >& bottom, const vector<Tensor<dtype>* >& top){};
	virtual void Backward(const vector<Tensor<dtype>* >& top, const vector<Tensor<dtype>* >& bottom){};
protected:
	void Init(){
		_weights.resize(2);
		//init weight
		vector<int> shape(4);
		shape[0] = 1;
		shape[1] = _num_output;
		shape[2] = _kernel_size == 0 ? _kernel_h : _kernel_size;
		shape[3] = _kernel_size == 0 ? _kernel_w : _kernel_size;
		_weights[0] = Tensor<dtype>(shape);
		wcv::randn(_weight[0], 0.F, 1.F);
		//init bias
		shape[0] = 1;
		shape[1] = _num_output;
		shape[2] = 1;
		shape[3] = 1;
		_weights[1] = Tensor<dtype>(shape);
		wcv::fill(_weights[1], 0);
	}
	void calc_output_shape(const vector<int>& _In,vector<int>& _Out){
		assert(_In.size() == 4);
		_Out.resize(4);
		_Out[0] = _In[0];
		_Out[1] = _num_output;
		_Out[2] = (_In[2] + 2 * _pad - _kernel_h) / _stride + 1;
		_Out[3] = (_In[3] + 2 * _pad - _kernel_w) / _stride + 1;
	}
private:
	int _num_output;
	int _kernel_size;
	int _kernel_h;
	int _kernel_w;
	int _stride;
	int _pad;
};
