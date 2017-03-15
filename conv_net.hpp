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

#define USE_SSE
#ifdef USE_SSE
#include <emmintrin.h>
#endif

template<typename dtype> class Tensor;
namespace wcv {
	template<typename dtype>
	float randn(float mean = 0.f, float sigma = 1.f) {
		std::default_random_engine re(time(NULL));
		std::normal_distribution<float> d(mean, sigma);
		return d(re);
	}
	template<typename dtype>
	void randn(Tensor<dtype>& _Out, float mean = 0.f, float sigma = 1.f) {
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
	void rand(Tensor<dtype>& _Out, float _Min = 0.f, float _Max = 1.f) {
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
	void fill(Tensor<dtype>& _Out, float _Val = 0.f) {
		size_t _Size = _Out.count();
		dtype* pdata = _Out.mutable_data();
		for (size_t i = 0; i < _Size; ++i) {
			pdata[i] = static_cast<dtype>(_Val);
		}
	}
	template<typename dtype>
	/**@brief _Out = alpha * A.t x B + beta * C
	*/
	void gemm(const Tensor<dtype>& _A, const Tensor<dtype>& _B, float alpha,
		const Tensor<dtype>& _C, float beta, Tensor<dtype>& _Out) {
		assert(_A.num() == _B.num() &&
			_A.channels() == _B.channels() &&
			_A.width() == _B.height());
		int N = _A.num();
		int C = _A.channels();
		int Ah = _A.height();
		int Aw = _A.height();
		int Bw = _B.width();

		if (!_C.empty()) {
			assert(_C.num() == _A.num() &&
				_C.channels() == _A.channels() &&
				_C.height() == Ah &&
				_C.width() == Bw);
		}

		float alpha = 1.F, beta = 0.F;
		vector<int> shape(4, 0);
		shape[0] = N, shape[1] = C;
		shape[2] = Ah, shape[3] = Bw;
		_Out.release();
		_Out = Tensor<dtype>(shape);
		for (size_t n = 0; n < N; ++n) {
			for (size_t c = 0; c < C; ++c) {
				dtype* ptrA = _A.offset(n, c);
				dtype* ptrB = _B.offset(n, c);
				dtype* ptrO = _Out.offset(n, c);
				dtype* ptrC = nullptr;
				if (!_C.empty()) {
					ptrC = _C.offset(n, c);
				}
				tensor_gemm<dtype>(Ah, Aw, Bw, ptrA, ptrB, alpha, ptrC, beta, ptrO);
			}
		}
	}
	template<typename dtype>
	/**@brief _Out = alpha * A.t x B + beta * C
	A = Matrix(M,N),B = Matrix(N,K),C = Matrix(M,K)
	*/
	void elemwise_add(int M, int N, int K, const dtype* A, const dtype* B, float alpha,
		const dtype* C, float beta, dtype*& _Out) {
		assert(A != nullptr && B != nullptr && _Out != nullptr);
		for (size_t m = 0; m < M; ++m) {
			for (size_t k = 0; k < K; ++k) {
				for (size_t n = 0; n < N; ++n) {
					_Out[m * N + k] +=
						A[m * N + n] * B[n * K + k];
					if (C != nullptr)
						_Out[m * N + k] += beta * C[m * N + k];
				}
			}
		}
	}
	template<typename dtype>
	/**@brief _Out = A(M,N) + msk_binary(M,N) * alpha * B(M,N)  */
	void gema(int M, int N, const dtype* A, const dtype* B, float alpha, const dtype* msk, dtype*& _Out) {
		size_t nCount = M * N, i = 0;
#ifdef USE_SSE
		if (std::is_same<typename std::decay<dtype>::type, float>::value &&
			nCount > 4) {
			__m128 a, b, c, t;
			t = _mm_set1_ps(alpha);
			for (i = 0; i < nCount - 4; i += 4) {
				a = _mm_loadu_ps((float*)(A + i));
				b = _mm_loadu_ps((float*)(B + i));
				if (msk != nullptr) {
					c = _mm_loadu_ps((float*)(msk + i));
					c = _mm_and_ps(b, c);//mask 1,0,1..
					b = _mm_mul_ps(_mm_mul_ps(b, c), t);
				}
				_mm_storeu_ps((float*)(_Out + i), _mm_add_ps(a, b));
			}
		}
#endif
		for (; i < nCount; i++) {
			int m = 1;
			if (msk != nullptr) {
				m = (msk[i] == 0);
			}
			_Out[i] = A[i] + m * alpha * B[i];
		}
		return result;
	}
	inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
		return static_cast<unsigned>(a) < static_cast<unsigned>(b);
	}
	template<typename dtype>
	/**reference from caffe*/
	void im2col(const dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, dtype* data_col) {
		assert(data_col != nullptr);
		const int output_h = (height + 2 * pad_h -
			kernel_h) / stride_h + 1;
		const int output_w = (width + 2 * pad_w -
			kernel_w) / stride_w + 1;
		const int channel_size = height * width;
		for (int channel = channels; channel--; data_im += channel_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					int input_row = -pad_h + kernel_row;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
							for (int output_cols = output_w; output_cols; output_cols--) {
								*(data_col++) = 0;
							}
						}
						else {
							int input_col = -pad_w + kernel_col;
							for (int output_col = output_w; output_col; output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
									*(data_col++) = data_im[input_row * width + input_col];
								}
								else {
									*(data_col++) = 0;
								}
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}
		}
	}
	template<typename dtype>
	/**reference from caffe*/
	void col2im(const dtype* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		dtype* data_im) {
		memset(data_im,0, height * width * channels*sizeof(dtype))
		const int output_h = (height + 2 * pad_h -
			kernel_h + 1)) / stride_h + 1;
		const int output_w = (width + 2 * pad_w -
			kernel_w + 1)) / stride_w + 1;
		const int channel_size = height * width;
		for (int channel = channels; channel--; data_im += channel_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					int input_row = -pad_h + kernel_row;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
							data_col += output_w;
						}
						else {
							int input_col = -pad_w + kernel_col;
							for (int output_col = output_w; output_col; output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
									data_im[input_row * width + input_col] += *data_col;
								}
								data_col++;
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}
		}
	}
	template<typename dtype>
	/**@brief _Out = (A^alpha) .* (B^beta) */
	void elemwise_mult(const int M, const int N, const dtype* A, const dtype* B, dtype*& _Out) {
		assert(A != nullptr && B != nullptr);
		size_t nCount = M * N,i = 0;
#ifdef USE_SSE
		if (std::is_same<typename std::decay<dtype>::type, float>::value &&
			nCount > 4) {
			__m128 s, a, b;
			for ( i = 0; i < nCount - 4; i +=4 ) {
				a = _mm_loadu_ps((float*)(A + i));
				b = _mm_loadu_ps((float*)(B + i));
				_mm_storeu_ps((float*)(_Out + i), _mm_mul_ps(a, b));
			}
		}
#endif // 
		for (; i < nCount; ++i)	{
			_Out[i] = A[i] * B[i];
		}
	}
}

#define TSR_MIN_DIM 4
#define TSR_CHANNEL_AXIS 1
template<typename dtype = float>
class Tensor
{
public:
	Tensor() :_shape(), _vdata() {};
	Tensor(const vector<int> dim, const dtype* data = nullptr) {
		assert(dim.size() >= TENSOR_MIN_DIM);
		_shape = _dim;
		_vdata.resize(dim(), 0);
		if (data != nullptr) {
			for (size_t i = 0; i < count(); ++i) {
				_vdata[i] = data[i];
			}
		}
	}
#ifdef HAVE_OPENCV
	Tensor(const vector<cv::Mat>& mvs) {
		_shape.resize(4);
		_shape[0] = mvs.size();
		_shape[1] = mvs[0].channels();
		_shape[2] = mvs[0].rows;
		_shape[3] = mvs[0].cols;
		for (size_t n = 0; n < _shape[0]; ++n) {
			cv::Mat& m = mvs[n];
			vector<cv::Mat> vm;
			if (m.channels() > 1) {
				cv::split(m, vm);
			}
			else {
				vm.push_back(m);
			}
			for (size_t c = 0; c < _shape[1]; ++c) {
				vector<dtype> vd = mat2dtype(vm[c]);
				_vdata.insert(_vdata.end(), vd.begin(), vd.end());
			}
		}
	}
	vector<dtype> mat2dtype(const cv::Mat& mat) {
		assert(mat.type() == CV_32FC1);
		std::vector<float> array;
		if (mat.isContinuous()) {
			if (mat.type() == CV_32FC1)
				array.assign((float*)mat.datastart,
				(float*)mat.dataend);
		}
		else {
			for (int i = 0; i < mat.rows; ++i) {
				if (mat.type() == CV_32FC1)
					array.insert(array.end(), (float*)mat.ptr<uchar>(i),
					(float*)mat.ptr<uchar>(i) + mat.cols);
			}
		}
		std::vector<float> arrayd(array.size());
		for (size_t i = 0; i < array.size(); i++) {
			arrayd[i] = static_cast<dtype>(array[i]);
		}
	}
#endif
	~Tensor() { release(); };
	void reshape(const vector<int>& _dim) {
		assert(dim.size() >= TENSOR_MIN_DIM);
		_shape = _dim;
		_vdata.resize(dim());
	}
	void reshape_likes(const Tensor& rhs) {
		_shape = rhs._shape;
		_vdata.resize(rhs._vdata.size());
		//_vdiff.resize(rhs._vdiff.size());
	}
	const dtype* const_data() {
		return _vdata.data();
	}
	dtype* mutable_data() {
		return const_cast<dtype*>(_vdata.data());
	}
	//const dtype* const_diff(){
	//	return _vdiff.data();
	//}
	//dtype* mutable_diff(){
	//	return const_cast<dtype*>(_vdiff.data());
	//}
	dtype* offset(int n = 0, int c = 0, int h = 0, int w = 0) {
		return const_cast<dtype*>(_vdata.data() + ((n*channels() + c)*height() + h)*width() + w);
	}
	void append(const Tensor& rhs) {
		assert(_shape.size() == rhs.size());
		bool bsm = true;
		for (size_t i = TSR_CHANNEL_AXIS; i < _shape.size(); ++i) {
			if (_shape[i] != rhs._shape[i]) {
				bsm = false; break;
			}
		}
		assert(bsm == true);
		_shape[0] += rhs._shape[0];
		_vdata.insert(_vdata.end(), rhs._vdata.begin(), rhs._vdata.end());
	}
	dtype& at(int n = 0, int c = 0, int h = 0, int w = 0) const {
		return _vdata[((n*channels() + c)*height() + h)*width() + w];
	}
	vector<int>& shape() const {
		return _shape;
	}
	size_t count(int startAxies = 0) {
		assert(!_shape.empty());
		size_t num_spatial_dim = 1;
		for (size_t i = startAxies; i < _shape.size(); i++) {
			num_spatial_dim *= _shape[i];
		}
		return num_spatial_dim;
	}
	int num() {
		assert(!_shape.empty() && _shape.size() > 1);
		return _shape[0];
	}
	int channels() {
		assert(!_shape.empty() && _shape.size() > 2);
		return _shape[1];
	}
	int height() {
		assert(!_shape.empty() && _shape.size() > 3);
		return _shape[2];
	}
	int width() {
		assert(!_shape.empty() && _shape.size() > 4);
		return _shape[3];
	}
	void release() {
		_shape.clear();
		_vdata.clear();
	}
	bool empty() {
		return _shape.empty() ||
			_vdata.empty();
	}
	Tensor& operator =(const Tensor& rhs) {
		this->_shape = rhs._shape;
		this->_vdata = rhs._vdata;
	}
private:
	vector<int> _shape;
	vector<dtype> _vdata;
	//vector<dtype> _vdiff;
};

class LayerParam
{
public:
	LayerParam() :_num_output(0) {};
	LayerParam(int num_output) :_num_output(num_output) {};
	~LayerParam() {};
protected:
	int _num_output;
};

class ConvParam :public LayerParam
{
public:
	ConvParam() :LayerParam(0), _kernel_size(0), _kernel_h(0),
		_kernel_w(0), _stride_w(0), _stride_h(0), _pad_h(0), _pad_w(0) {};
	ConvParam(int num_output, int kernel_size, int stride, int pad) :
		LayerParam::LayerParam(num_output), _kernel_size(kernel_size),
		_kernel_h(kernel_size), _kernel_w(kernel_size),
		_stride_w(stride), _stride_h(stride), _pad_w(pad), _pad_h(pad) {};
	ConvParam(int num_output, int kernel_h, int kernel_w,
		int stride_h, int stride_w, int pad_h, int pad_w) :
		LayerParam::LayerParam(num_output), _kernel_size(0),
		_kernel_h(kernel_h), _kernel_w(kernel_w),
		_stride_h(stride_h), _stride_w(stride_w),
		_pad_h(pad_h), _pad_w(pad_w) {};
	~ConvParam() {};
private:
	int _kernel_size;
	int _kernel_h;
	int _kernel_w;
	int _stride_h;
	int _stride_w;
	int _pad_h;
	int _pad_w;
};

template<typename dtype>
class Layer
{
public:
	Layer() {};
	~Layer() {};
	virtual void Reshape(const vector<Tensor<dtype>* >& bottom,
		const vector<Tensor<dtype>* >& top) = 0;
	virtual void Forward(const vector<Tensor<dtype>* >& bottom,
		const vector<Tensor<dtype>* >& top) = 0;
	virtual void Backward(const vector<Tensor<dtype>* >& top,
		const vector<Tensor<dtype>* >& bottom) = 0;
protected:
	void Init() {};
protected:
	vector<Tensor<dtype> > _diff;
	vector<Tensor<dtype> > _data;
	//_weights[0]->kernel weight ,_weights[1]->bias
	vector<Tensor<dtype> > _weights;
};

template<typename dtype>
class ConvLayer :public Layer<dtype>
{
public:
	ConvLayer() :_diff(), _data() {};
	ConvLayer(LayerParam& _Param) {
		ConvParam* convParam = dynamic_cast<ConvParam*>(&_Param);
		assert(convParam != nullptr);
		int kernel_h = convParam->_kernel_size > 0 ?
			convParam->_kernel_size : convParam->_kernel_h;
		int kernel_w = convParam->_kernel_size > 0 ?
			convParam->_kernel_size : convParam->_kernel_w;
		InitParam(convParam->_num_output, kernel_h,
			kernel_w, convParam->_stride_h, convParam->_stride_w,
			convParam->_pad_h, convParam->_pad_w);
		Init();
	}
	ConvLayer(int num_output, int kernel_size,
		int stride, int pad) {
		InitParam(num_output, kernel_size,
			kernel_size, stride, stride, pad, pad);
		Init();
	}
	ConvLayer(int num_output, int kernel_h, int kernel_w,
		int stride_h, int stride_w, int pad_h, int pad_w) {
		InitParam(num_output, kernel_h, kernel_w,
			stride_h, stride_w, pad_h, pad_w);
		Init();
	}
	~ConvLayer() {};
	virtual void Reshape(const vector<Tensor<dtype>* >& bottom,
		const vector<Tensor<dtype>* >& top) {
		_diff.resize(bottom->size());
		_data.resize(bottom->size());
		vector<int> input_shape = bottom[0]->shape();
		vector<int> output_shape;
		calc_output_shape(input_shape, output_shape);
		for (size_t i = 0; i < top.size(); ++i) {
			top[i] = Tensor<dtype>(output_shape);
			_diff[i] = Tensor<dtype>(output_shape);
			_data[i] = Tensor<dtype>(output_shape);
		}
		output_shape[3] = 1; //n*c*h*1
		_bias_multiplier.reshape(output_shape);
	}
	virtual void Forward(const vector<Tensor<dtype>* >& bottom,
		const vector<Tensor<dtype>* >& top) {
		float* weight = _weights[0].mutable_data();
		for (size_t i = 0; i < bottom.size(); ++i) {
			Tensor<dtype> col_buff;
			tensor_im2col(*bottom[i], col_buff);
			int weight_dims = _weights[0].count(2);
			for (size_t n = 0; n < bottom[i]->shape()[0]; ++n) {
				for (size_t k = 0; k < _num_output; ++k) {
					for (size_t c = 0; c < col_buff.shape()[1]; ++c) {
						wcv::gemm(col_buff.shape()[2], col_buff.shape()[3], (dtype)1,
							col_buff.offset(n, c), weight + k * weight_dim,
							(dtype)1, top[i]->offset(n, c), (dtype)1, top[i]->offset(n, k));
					}
				}
			}
			//bias
			wcv::elemwise_add(top[i]->shape()[2], top[i]->shape[3],
				top[i]->const_data(), _weights[1].const_data(),
				1, nullptr, top[i]->mutable_data());
			_data[i] = *top[i];
		}
	}
	virtual void Backward(const vector<Tensor<dtype>* >& top,
		const vector<Tensor<dtype>* >& bottom) {
		assert(top[0]->count(0) == bottom[0]->count());
		for (size_t i = 0; i < top.size(); i++)	{
			for (size_t n = 0; n < top[i]->shape()[0]; ++n)	{
				for (size_t c = 0; c < top[i]->shape()[1]; ++c)	{
					wcv::elemwise_mult(top[i]->shape()[2], top[i]->shape()[3],
						top[i]->offset(n, c), _data[i].offset(n, c), bottom[i]->offset(n, c));
				}
			}
			wcv::elemwise_mult()
		}
	}
protected:
	void InitParam(int num_output, int kernel_h,
		int kernel_w, int stride_h, int stride_w,
		int pad_h, int pad_w) {
		assert(num_output != 0 && kernel_h != 0 &&
			kernel_w != 0 && stride_h != 0 &&
			stride_w != 0 && pad_h != 0 && pad_w != 0);
		_num_output = num_output;
		_kernel_size = kernel_size;
		_kernel_h = kernel_size;
		_kernel_w = kernel_size;
		_stride_h = stride_h;
		_stride_w = stride_w;
		_pad_h = pad_h;
		_pad_w = pad_w;
	}
	void Init() {
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
		wcv::fill(_weights[1], 1);
	}
	void calc_output_shape(const vector<int>& _In, vector<int>& _Out) {
		assert(_In.size() == 4);
		_Out.resize(4);
		_Out[0] = _In[0];
		_Out[1] = _num_output;
		_Out[2] = (_In[2] + 2 * _pad_h - _kernel_h) / _stride_h + 1;
		_Out[3] = (_In[3] + 2 * _pad_w - _kernel_w) / _stride_w + 1;
	}
	void tensor_im2col(const Tensor<dtype>& _In, Tensor<dtype>& _Out) {
		int conv_out_h = (_In.shape()[2] + 2 * _pad_h - _kernel_h) / _stride_h + 1;
		int conv_out_w = (_In.shape()[3] + 2 * _pad_w - _kernel_w) / _stride_w + 1;
		int num_spatial_dim = conv_out_h * conv_out_w;
		vector<int> shape(4, 0);
		shape[0] = _In.shape()[0];
		shape[1] = _In.shape()[1];
		shape[2] = num_spatial_dim;
		shape[3] = _kernel_h * _kernel_w;
		_Out.release();
		_Out = Tensor<dtype>(shape);
		wcv::im2col(_In.const_data(), _In.shape()[1], _In.shape()[2], _In.shape()[3],
			_kernel_h, _kernel_w, _pad_h, _pad_w, _stride_h, _stride_w, _Out.mutable_data());
	}
private:
	int _num_output;
	int _kernel_size;
	int _kernel_h;
	int _kernel_w;
	int _stride_h;
	int _stride_w
		int _pad_h;
	int _pad_w;
	Tensor<dtype> _bias_multiplier;
};
