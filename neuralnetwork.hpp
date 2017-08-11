#pragma once
#include <algorithm>
#include <vector>
#include <assert.h>
#include <iomanip>
#define HAVE_OPENCV
#if defined HAVE_OPENCV
#include <opencv2/core/core.hpp>
#endif

using namespace std;

namespace wcv{

	template <typename _Tp = int>
	class Range_
	{
	public:
		Range_() :start(0), end(0){};
		Range_(int start, int end) :
			start(start), end(end){}
		~Range_(){};
	public:
		_Tp start;
		_Tp end;
	};

	template<typename _Type = float>
	class Matrix
	{
	public:
		Matrix() :rows(0), cols(0){};
		Matrix(int rows, int cols) :
			rows(rows), cols(cols)
		{
			_matrix.resize(rows);
			for (size_t i = 0; i < rows; i++) {
				_matrix[i].resize(cols);
			}
		}
		Matrix(int rows, int cols, void* data) :
			rows(rows), cols(cols){
			_matrix.resize(rows);
			for (size_t i = 0; i < rows; i++) {
				_matrix[i].resize(cols);
				for (size_t j = 0; j < _matrix[i].size(); j++) {
					_matrix[i][j] = static_cast<_Type>
						(*(data + i * cols + j));
				}
			}
		}

		Matrix(int rows, int cols, vector<_Type> vec) :
			rows(rows), cols(cols)
		{
			assert(rows * cols == vec.size());
			_matrix.resize(rows);
			for (size_t i = 0; i < rows; i++) {
				_matrix[i].resize(cols);
				for (size_t j = 0; j < _matrix[i].size(); j++) {
					_matrix[i][j] = static_cast<_Type>(vec[i * cols + j]);
				}
			}
		}
#if defined HAVE_OPENCV
		Matrix(const cv::Mat& _cvmat):
			rows(_cvmat.rows), cols(_cvmat.cols * _cvmat.channels())
		{
			int type = 0;
			if (typeid(_Type) == typeid(double)){
				type = CV_64F;
			}
			else if (typeid(_Type) == typeid(uchar)){
				type = CV_8U;
			}
			else if (typeid(_Type) == typeid(float)){
				type = CV_32F;
			}
			else if (typeid(_Type) == typeid(int)){
				type = CV_16U;
			} else{
				type = CV_8U;
			}
			CV_Assert(_cvmat.type() == type);

			_matrix.resize(rows);
			int channels = _cvmat.channels();
			for (size_t i = 0; i < _cvmat.rows; i++) {
				int idx = 0;
				_matrix[i].resize(cols);
				_Type* p = (_Type*)_cvmat.ptr<_Type>(i);
				for (size_t j = 0; j < _cvmat.cols; j++){
					for (size_t k = 0; k < channels; k++) {
						_matrix[i][idx++] = p[j * channels + k];
					}
				}
			}
		}

		static cv::Mat Matrix2CVMat(const Matrix<_Type>& m){
			int type = 0;
			if (typeid(_Type) == typeid(double)){
				type = CV_64FC1;
			} else if (typeid(_Type) == typeid(uchar)){
				type = CV_8UC1;
			} else if (typeid(_Type) == typeid(float)){
				type = CV_32FC1;
			} else if (typeid(_Type) == typeid(int)){
				type = CV_16UC1;
			}
			cv::Mat im(m.rows, m.cols, type);
			for (size_t i = 0; i < rows; i++) {
				_Type* p = (_Type*)im.ptr<_Type>(i);
				for (size_t j = 0; j < cols; j++){
					p[j] = _matrix[i][j];
				}
			}
			return im;
		}
#endif
		~Matrix(){ clear(); };

		Matrix subMat(Range_<int> rowRange, Range_<int> colRange){
			assert(rowRange.end >= rowRange.start);
			assert(colRange.end >= colRange.start);
			Matrix submat(rowRange.end - rowRange.start,
				colRange.end - colRange.start);
			for (size_t i = rowRange.start; i < rowRange.end; i++) {
				for (size_t j = colRange.start; j < colRange.end; j++) {
					submat._matrix[i - rowRange.start][j - colRange.start] =
						_matrix[i][j];
				}
			}
			return submat;
		}

		Matrix rowRange(Range_<int> rowRange) {
			assert(rowRange.end >= rowRange.start);
			return subMat(rowRange, Range_<int>(0, cols));
		}

		Matrix colRange(Range_<int> colRange) {
			assert(colRange.end >= colRange.start);
			return subMat(Range_<int>(0, rows), colRange);
		}

		Matrix row(int row) {
			assert(0 <= row && row < rows);
			return subMat(Range_<int>(row, row + 1), Range_<int>(0, cols));
		}

		Matrix col(int col) {
			assert(0 <= col && col < cols);
			return subMat(Range_<int>(0, rows), Range_<int>(col, col + 1));
		}

		vector<vector<_Type> >& toVector(){
			return _matrix;
		}

		_Type& at(int row, int col) {
			assert(0 <= row && row < rows);
			assert(0 <= col && col < cols);
			return _matrix[row][col];
		}

		void copyTo(Matrix& m, Range_<int> row_range,
			Range_<int> col_range){
			assert(row_range.start >= 0 && row_range.end <= m.rows);
			assert(col_range.start >= 0 && col_range.end <= m.cols);
			assert((row_range.end - row_range.start) == rows &&
				(col_range.end - col_range.start) == cols);
			for (size_t i = row_range.start; i < row_range.end; i++) {
				for (size_t j = col_range.start; j < col_range.end; j++) {
					m.at(i,j) = _matrix[i - row_range.start][j - col_range.start];
				}
			}
		}

		void push_back(const Matrix& mat){
			if (!_matrix.empty())
				assert(cols == mat.cols);
			else
				cols = mat.cols;
			rows += mat.rows;
			for (size_t i = 0; i < mat.rows; i++) {
				_matrix.push_back(mat._matrix[i]);
			}
		}

		void clear(){
			rows = 0;
			cols = 0;
			_matrix.clear();
		}

		void setData(int row, int col, _Type data){
			_matrix[row][col] = data;
		}

		Matrix t(){
			Matrix t(cols, rows);
			for (size_t i = 0; i < t.rows; i++) {
				for (size_t j = 0; j < t.cols; j++){
					t._matrix[i][j] = _matrix[j][i];
				}
			}
			return t;
		}

		static Matrix zeros(int rows, int cols){
			Matrix mat(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++) {
					mat.setData(i, j, (_Type)0);
				}
			}
			return mat;
		}

		static Matrix ones(int rows, int cols){
			Matrix mat(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++) {
					mat.setData(i, j, (_Type)1);
				}
			}
			return mat;
		}

		static Matrix shuffle(const Matrix& m_){
			Matrix m = m_;
			std::random_shuffle(m._matrix.begin(), m._matrix.end(),
				[](int i)->int{
				return std::rand() % i;
			});
			return m;
		}

		static Matrix merge(const Matrix& m0, const Matrix& m1){
			assert(m0.rows == m1.rows);
			Matrix mat(m0.rows, m0.cols + m1.cols);
			for (size_t i = 0; i < mat.rows; i++) {
				for (size_t j = 0; j < mat.cols; j++){
					if (j < m0.cols){
						mat._matrix[i][j] = m0._matrix[i][j];
					}
					else{
						mat._matrix[i][j] = m1._matrix[i][j - m0.cols];
					}
				}
			}
			return mat;
		}

		static Matrix rand(int rows, int cols){
			Matrix result = Matrix(rows, cols);
			for (size_t i = 0; i < result.rows; i++) {
				for (size_t j = 0; j < result.cols; j++){
					result.at(i,j) = std::rand() / (RAND_MAX + 1.0);
				}
			}
			return result;
		}

		static Matrix randn(int rows, int cols,float esp = 1){
			Matrix result = Matrix(rows, cols);
			for (size_t i = 0; i < result.rows; i++) {
				for (size_t j = 0; j < result.cols; j++){
					result.at(i, j) = _randn(-esp, esp);
				}
			}
			return result;
		}

		static Matrix fromString(const string& datastr){
			Matrix<_Type> m;
			if (datastr.length() > 3 &&
				string::npos != datastr.find("[") &&
				string::npos != datastr.find("]"))
			{
				vector<string> vec;
				string data = datastr.substr(1, datastr.size() - 3);
				strSplit(data, ";", vec);
				int rows = 0, cols = 0;
				vector<_Type> datum;
				rows = vec.size();
				for (size_t i = 0; i < vec.size(); i++)	{
					string s = vec[i];
					//if (string::npos != s.find(";"))
					//	s = s.substr(0, s.size() - 1);
					vector<string> tmp;
					strSplit(s, ",", tmp);
					cols = tmp.size();
					for (size_t j = 0; j < tmp.size(); j++)	{
						_Type val = (_Type)atof(tmp[j].c_str());
						datum.push_back(val);
					}
				}
				m = Matrix<_Type>(rows, cols, datum);
			}
			return m;
		}

		string toString(bool singleLine = true){
			stringstream buff;
			buff << "[";
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++) {
					if (j < cols - 1)
						buff << _matrix[i][j] << ",";
					else
						buff << _matrix[i][j] << ";";
				}
				if (!singleLine && i < rows - 1)
					buff << endl;
				else if (i == rows - 1)
					buff << "]";
			}
			return buff.str();
		}

		_Type dot(const Matrix& m){
			assert(rows == m.rows);
			assert(cols == m.cols);
			_Type result = static_cast<_Type>(0);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++){
					result += _matrix[i][j] * m._matrix[i][j];
				}
			}
			return result;
		}

		Matrix hadamardProduct(const Matrix& mat){
			assert(rows == mat.rows);
			assert(cols == mat.cols);
			Matrix result(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++) {
					result._matrix[i][j] = _matrix[i][j] *
						mat._matrix[i][j];
				}
			}
			return result;
		}

		Matrix operator+ (const Matrix& rhs){
			assert(rows == rhs.rows);
			assert(cols == rhs.cols);
			Matrix result(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++){
					result._matrix[i][j] = _matrix[i][j] +
						rhs._matrix[i][j];
				}
			}
			return result;
		}

		Matrix operator+ (const _Type scalar){
			Matrix result(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++){
					result[i][j] = _matrix[i][j] + scalar;
				}
			}
			return result;
		}

		Matrix operator- (const Matrix& rhs){
			assert(rows == rhs.rows);
			assert(cols == rhs.cols);
			Matrix result(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++){
					result._matrix[i][j] = _matrix[i][j] -
						rhs._matrix[i][j];
				}
			}
			return result;
		}

		Matrix operator- (const _Type scalar){
			Matrix result(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++){
					result._matrix[i][j] = _matrix[i][j] - scalar;
				}
			}
			return result;
		}

		Matrix operator / (const _Type scalar){
			Matrix result(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++){
					result._matrix[i][j] = _matrix[i][j] / scalar;
				}
			}
			return result;
		}

		Matrix operator * (const _Type scalar){
			Matrix result(rows, cols);
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++){
					result._matrix[i][j] = _matrix[i][j] * scalar;
				}
			}
			return result;
		}

		Matrix operator *(const Matrix& rhs){
			_Type const ZERO = static_cast<_Type>(0);
			assert(cols == rhs.rows);
			int newRows = rows;
			int newCols = rhs.cols;
			Matrix result(newRows, newCols);

			for (size_t i = 0; i < newRows; ++i)
			for (size_t j = 0; j < newCols; ++j) {
				result._matrix[i][j] = ZERO;
				for (size_t index = 0; index < cols; ++index)
					result._matrix[i][j] +=
					_matrix[i][index]
					* rhs._matrix[index][j];
			}
			return result;
		}

		Matrix& operator =(const Matrix& rhs){
			clear();
			alloc(rhs.rows, rhs.cols);
			rows = rhs.rows;
			cols = rhs.cols;
			for (size_t i = 0; i < rhs.rows; i++) {
				for (size_t j = 0; j < rhs.cols; j++){
					_matrix[i][j] = rhs._matrix[i][j];
				}
			}
			return *this;
		}

		bool operator ==(const Matrix& rhs){
			bool bres = true;
			if (rows == rhs.rows && cols == rhs.cols){
				for (size_t i = 0; i < rows; i++) {
					for (size_t j = 0; j < cols; j++) {
						if (_matrix[i][j] != rhs._matrix[i][j])
							bres = false;
					}
				}
			}
			else{
				bres = false;
			}
			return bres;
		}

		bool operator !=(const Matrix& rhs){
			return !(*this == rhs);
		}

		vector<_Type>& operator [](int idx){
			assert(idx >= 0 && idx < rows);
			return _matrix[idx];
		}

		friend ostream& operator <<(ostream& os, const Matrix& rhs){
			os.setf(ios::fixed);
			os.precision(8);
			os << "[";
			for (size_t i = 0; i < rhs.rows; i++) {
				for (size_t j = 0; j < rhs.cols; j++) {
					if (j < rhs.cols - 1)
						os << rhs._matrix[i][j] << ",";
					else
						os << rhs._matrix[i][j] << ";";
				}
				if (i < rhs.rows - 1)
					os << endl;
				else
					os << "]";
			}
			return os;
		}
	public:
		int rows;
		int cols;
	private:
		void alloc(int rows, int cols){
			assert(rows >= 0 && cols >= 0);
			_matrix.resize(rows);
			for (size_t i = 0; i < rows; i++) {
				_matrix[i].resize(cols);
			}
		}

		static _Type _randn(_Type min, _Type max) {
			return min + (max - min)*std::rand() / (RAND_MAX + 1.0);
		}
	private:
		typedef vector<vector<_Type> > MAT;
		MAT _matrix;
	};

#define PI 3.1415926535898
	Matrix<float> gaussian2D(int ksize_x, int ksize_y, float sigma = 1.f){
		Matrix<float> kernel(ksize_y, ksize_x);
		int x0 = ksize_x / 2;
		int y0 = ksize_y / 2;
		float A = 1. / (sigma * sqrt(2 * PI));
		for (size_t i = 0; i < ksize_y; i++) {
			for (size_t j = 0; j < ksize_x; j++) {
				kernel.at(i, j) = A * exp(-((i - x0)*(i - x0) / (2 * sigma*sigma) +
					(j - y0)*(j - y0) / (2 * sigma*sigma)));
			}
		}
		return kernel;
	}


	template<typename _Tp = float>
	class Activation
	{
	public:
		Activation();
		~Activation();
#if 1
		static _Tp activation(_Tp value) {
			return 1. / (1 + exp(-value));
		}

		static _Tp derivation(_Tp value) {
			_Tp val = activation(value);
			return val * (1 - val);
		}
#else
		static _Tp activation(_Tp value) {
			return (1. - exp(-2*value)) / (1. + exp(- 2 * value));
		}

		static _Tp derivation(_Tp value) {
			_Tp val = activation(value);
			return 1. - val * val;
		}
#endif // 0


		static Matrix<_Tp> activation(Matrix<_Tp>& m){
			Matrix<_Tp> result(m.rows, m.cols);
			for (size_t i = 0; i < m.rows; i++) {
				for (size_t j = 0; j < m.cols; j++) {
					_Tp value = m.at(i, j);
					result.at(i, j) = activation(value);
				}
			}
			return result;
		}

		static Matrix<_Tp> derivation(Matrix<_Tp>& m){
			Matrix<_Tp> result(m.rows, m.cols);
			for (size_t i = 0; i < m.rows; i++) {
				for (size_t j = 0; j < m.cols; j++) {
					_Tp value = m.at(i, j);
					result.at(i, j) = derivation(value);
				}
			}
			return result;
		}
	};

	template <typename _Tp,
		typename _activation_>
	class Layer
	{
	public:
		Layer(int input, int output) :
			nOutputs(output), nInputs(input){
			initWeights();
		}
		~Layer(){
			_weight_and_bias.clear();
			_act_data.clear();
		}

		Matrix<_Tp>& getWeights(){
			return _weight_and_bias;
		}

		void setWeights(const Matrix<_Tp>& weight_and_bias){
			_weight_and_bias = weight_and_bias;
		}

		Matrix<_Tp> getLossErr(){
			return _loss_err;
		}

		Matrix<_Tp> getZData(){
			return _z_data;
		}

		void setLossErr(const Matrix<_Tp>& loss_err){
			_loss_err = loss_err;
		}

		Matrix<_Tp>& getActData(){
			return _act_data;
		}

		void forward(const Matrix<_Tp>& bottom_, Matrix<_Tp>& top) {
			Matrix<_Tp>  bottom = bottom_;
			top.clear();
			Matrix<_Tp> bias;
			Matrix<_Tp> weights = _weight_and_bias.subMat(
				Range_<int>(0, _weight_and_bias.rows),
				Range_<int>(0, _weight_and_bias.cols - 1));
			Matrix<_Tp> b = _weight_and_bias.col(_weight_and_bias.cols - 1).t();
			for (size_t i = 0; i < bottom.rows; i++) {
				bias.push_back(b);
			}
			_z_data = bottom * weights.t() + bias;
			top = _activation_::activation(_z_data);
			_act_data = top;
		}

		void backward(Matrix<_Tp>& top_weight, Matrix<_Tp>& top_err){
			//assert(top_weight.rows == top_err.rows);
			Matrix<_Tp> weights = top_weight.subMat(
				Range_<int>(0, top_weight.rows),
				Range_<int>(0, top_weight.cols - 1));
			Matrix<_Tp> temp = top_err * weights;
			Matrix<_Tp> deri = _activation_::derivation(_z_data);
			_loss_err = temp.hadamardProduct(deri);
		}

		void update(Matrix<_Tp>& actives_, float lr,float lamda){
			//update weights
			assert(actives_.rows == _loss_err.rows);
			Matrix<_Tp> weights = _weight_and_bias.subMat(
				Range_<int>(0, _weight_and_bias.rows),
				Range_<int>(0, _weight_and_bias.cols - 1));
			Matrix<_Tp> bias = _weight_and_bias.col(_weight_and_bias.cols - 1);
			Matrix<_Tp> deltaW = Matrix<_Tp>(_loss_err.cols, weights.cols);
			Matrix<_Tp> deltaB = Matrix<_Tp>(weights.rows, 1);
			for (size_t i = 0; i < actives_.rows; i++) {
				Matrix<_Tp> err = _loss_err.row(i);
				Matrix<_Tp> active = actives_.row(i);
				deltaB = deltaB + err.t();
				deltaW = deltaW + err.t() * active;
			}
			deltaW = deltaW / actives_.rows;
			deltaB = deltaB / actives_.rows;
			Matrix<_Tp> regu = weights * lamda;
			weights = weights - (deltaW + regu) * lr;  //regulization
			bias = bias - deltaB * lr;
			_weight_and_bias = Matrix<_Tp>::merge(weights, bias);
		}

		int nInputs;
		int nOutputs;

	private:
		void initWeights(){
			Matrix<_Tp> weights = Matrix<_Tp>::randn(nOutputs, nInputs,1.);
			Matrix<_Tp> bias = Matrix<_Tp>::ones(nOutputs, 1);
			_weight_and_bias = Matrix<_Tp>::merge(weights, bias);
		}

		void initWeights(Matrix<_Tp>& weights, Matrix<_Tp>& bias){
			_weight_and_bias = Matrix<_Tp>::merge(weights, bias);
		}

		Matrix<_Tp> calcGradApprox(){
			for (size_t i = 0; i < _weight_and_bias.rows; i++)
			{

			}
		}

		bool grad_check(const Matrix<_Tp>& gradSGD, const Matrix<_Tp>& gradApprox, float esp = 0.001){
			return true;
		}
	private:
		//|weight,..,bias|
		//|weight,..,bias|
		Matrix<_Tp> _weight_and_bias; //weight + bias
		Matrix<_Tp> _act_data;	//layer activation
		Matrix<_Tp> _z_data;
		Matrix<_Tp> _loss_err;	//layer error
	};

	template<typename _Tp,
		typename _activation>
	class Net
	{
	public:
		Net(){};
		~Net(){};

		class Param{
		public:
			int maxIter;
			float lr;
			float esp;
			float regu_lamda;
			int mini_batch;
			bool shuffle_data;
			Param(int maxIter_, float lr_ = 0.1, float esp_ = 0.001, float regu_lamda_ = 0.2,
				int mini_batch_ = 20, bool shuffle_data_ = true) :
				maxIter(maxIter_), lr(lr_), esp(esp_), regu_lamda(regu_lamda_),
				mini_batch(mini_batch_), shuffle_data(shuffle_data_)
			{
			}
			Param() :maxIter(0), lr(0), esp(0), regu_lamda(0),
				mini_batch(0), shuffle_data(false)
			{
			}
		};

		Matrix<int> getLayers(){
			int i = 0;
			Matrix<int> layers(1, _networks.size()+1);
			for (i = 0; i < _networks.size(); i++) {
				layers.at(0, i) = _networks[i].nInputs;
			}
			layers.at(0, i) = _networks[i-1].nOutputs;
			return layers;
		}

		int getNetSize(){
			return _networks.size();
		}

		Layer<_Tp, _activation> getLayer(int layerId){
			assert(layerId >= 0 && layerId < _networks.size());
			return _networks[layerId];
		}

		Matrix<_Tp> forward(const Matrix<_Tp>& data_){
			Matrix<_Tp> data = data_;
			assert(data.rows == 1 || data.cols == 1);
			if (data.rows > 1){
				data = data.t();
			}
			for (size_t j = 0; j < _networks.size(); j++) {
				Layer<_Tp, _activation>& layer = _networks[j];
				layer.forward(data, data);
			}
			return data;
		}

		vector<Matrix<_Tp> > forward(const vector<Matrix<_Tp> >& mv){
			vector<Matrix<_Tp> > responses;
			for (size_t i = 0; i < mv.size(); i++) {
				Matrix<_Tp> rsp = forward(mv[i]);
				responses.push_back(rsp);
			}
			return responses;
		}

		template<class _one_round_callback>
		void train(const Matrix<_Tp>& trainData, const Matrix<_Tp>& responses,
			Param& const param, _one_round_callback one_round_callback = [&](int iter, float err){})
		{
			assert(check_net());
			float error = FLT_MAX;
			int iter = 0;
			_lr = param.lr;
			_lamda = param.regu_lamda;
			Matrix<_Tp> dataMat =
				Matrix<_Tp>::merge(trainData, responses);
			vector<Matrix<_Tp> > data_batches;
			vector<Matrix<_Tp> > resp_batches;
			int nTrainDataCols = trainData.cols;
			while (iter < param.maxIter && error > param.esp) {
				data_batches.clear();
				resp_batches.clear();
				if (param.shuffle_data)
					dataMat = Matrix<_Tp>::shuffle(dataMat);
				int nSize = dataMat.rows / param.mini_batch;
				Matrix<_Tp> one_batch;
				for (size_t i = 0; i < nSize; i++) {
					if (i < nSize){
						one_batch = dataMat.rowRange(
							Range_<int>(i*param.mini_batch, (i + 1)*param.mini_batch));
					}

					if ((i == nSize - 1) && (dataMat.rows - i*param.mini_batch > 0)){
						one_batch = dataMat.rowRange(
							Range_<int>(i*param.mini_batch, dataMat.rows));
					}
					Matrix<_Tp> dat = one_batch.subMat(
						Range_<int>(0, one_batch.rows), Range_<int>(0, nTrainDataCols));
					Matrix<_Tp> rsp = one_batch.subMat(
						Range_<int>(0, one_batch.rows), Range_<int>(nTrainDataCols, one_batch.cols));
					data_batches.push_back(dat);
					resp_batches.push_back(rsp);
					error = solve_one_batch(dat, rsp);
					one_round_callback(iter++, error);
					if (error < param.esp)
						break;
				}
				//error = solve(data_batches, resp_batches);
				//one_round_callback(iter++, error);
			}
		}

		void add(const Layer<_Tp, _activation>& layer){
			_networks.push_back(layer);
		}

		Net<_Tp, _activation>& operator <<(const Layer<_Tp, _activation>& layer){
			if (!_networks.empty())
				assert(_networks[_networks.size() - 1].nOutputs == layer.nInputs);
			this->_networks.push_back(layer);
			return *this;
		}

		double calcError(const Matrix<_Tp>& samples_, const Matrix<_Tp>& responses_, Matrix<int>* err_grid) {
			Matrix<_Tp> samples = samples_;
			Matrix<_Tp> responses = responses_;
			if (samples.rows != responses.rows) {
				CV_Error(CV_StsError, "input samples and response not match.");
			}
			float nErrCount = 0;
			float nSamples = (float)samples.rows;
			Matrix<int> grid = Matrix<int>(responses.cols, responses.cols);

			for (size_t i = 0; i < samples.rows; i++) {
				Matrix<_Tp> s = samples.row(i);
				Matrix<_Tp> r = responses.row(i);
				Matrix<_Tp> rsp = this->forward(s);
				int raw_idx = std::max_element(r[0].begin(), r[0].end()) - r[0].begin();
				int rsp_idx = std::max_element(rsp[0].begin(), rsp[0].end()) - rsp[0].begin();
				if (rsp_idx != raw_idx) {
					nErrCount += 1;
				}
				int x = raw_idx;
				int y = rsp_idx;
				grid.at(y, x)++;
			}
			if (err_grid != NULL)
				*err_grid = grid;
			return nErrCount / nSamples;
		}

		Matrix<int> formatErrGrid(const Matrix<int>& err_grid_/*, ostream& os*/)
		{
			Matrix<int> err_grid = err_grid_;
			CV_Assert(err_grid.rows == err_grid.cols);
			int length = err_grid.rows;
			Matrix<int> row_lab = Matrix<int>(1, length);
			for (size_t i = 0; i < length; i++) {
				row_lab.at(0, i) = i;
			}
			Matrix<int> col_lab = row_lab.t();
			Matrix<int> formatterMat = Matrix<int>::
				zeros(err_grid.rows + 1, err_grid.cols + 1);
			row_lab.copyTo(formatterMat,Range_<int>(0,1),Range_<int>(1,formatterMat.cols));
			col_lab.copyTo(formatterMat,Range_<int>(1, formatterMat.rows), Range_<int>(0, 1));
			err_grid.copyTo(formatterMat,Range_<int>(1, formatterMat.rows), Range_<int>(1, formatterMat.cols));
			formatterMat.at(0, 0) = -1;
			//os << "grade grid=\n" << formatterMat << endl;
			return formatterMat;
		}

		void save(const string& fname) {
			ofstream os(fname,ios::out | ios::binary);
			stringstream buff;
			buff << this->getLayers().toString() << "|";
			for (size_t i = 0; i < _networks.size(); i++) {
				Layer<_Tp, _activation>& layer = _networks[i];
				if (i < _networks.size() - 1)
					buff << layer.getWeights().toString() << "|";
				else
					buff << layer.getWeights().toString();
			}
			os << buff.str();
			os.close();
		}

		void load(const string& fname) {
			ifstream fs(fname, ios::in | ios::binary);
			string buff = "",str="";
			while (getline(fs, str)){
				buff += str;
			}
			vector<string> vecStrs;
			strSplit(buff, "|", vecStrs);
			vector<Matrix<_Tp> > mvs;
			for (size_t i = 0; i < vecStrs.size(); i++)	{
				string& one_mat_str = vecStrs[i];
				Matrix<_Tp> m = Matrix<_Tp>::fromString(one_mat_str);
				mvs.push_back(m);
			}
			//restruct network
			if (mvs.empty()) return;
			Matrix<_Tp> layerMat = mvs[0];
			assert(layerMat.rows == 1 && layerMat.cols > 2);
			_networks.clear();
			for (size_t i = 0; i < layerMat.cols-1; i++)	{
				int nInput = layerMat.at(0, i);
				int nOutput = layerMat.at(0, i + 1);
				Layer<_Tp, _activation> layer(nInput, nOutput);
				layer.setWeights(mvs[i + 1]);
				*this << layer;
			}
			assert(check_net());
			fs.close();
		}
	protected:
		float solve(vector<Matrix<_Tp> >& datas,
			vector<Matrix<_Tp> >& responses) {
			assert(!datas.empty() && !responses.empty());
			assert(datas.size() == responses.size());
			assert(datas[0].rows == responses[0].rows);

			float error = FLT_MAX;
			for (size_t i = 0; i < datas.size(); i++) {
				Matrix<_Tp>& one_batch = datas[i];
				Matrix<_Tp>& one_batch_rsp = responses[i];
				Matrix<_Tp> fw_input = one_batch;
				//layers forward 
				for (size_t j = 0; j < _networks.size(); j++) {
					Layer<_Tp, _activation>& layer = _networks[j];
					layer.forward(fw_input, fw_input);
				}

				//calc last layer loss
				Matrix<_Tp> err_L = fw_input - one_batch_rsp;
				_networks[_networks.size() - 1].setLossErr(err_L);

				Matrix<_Tp> Htheta = fw_input;
				Matrix<_Tp> Jval = calc_cost_func_J(Htheta, one_batch_rsp,1);
				_Tp regu = regulization(_lamda);
				float sum = 0;
				for (size_t r = 0; r < Jval.rows; r++) {
					for (size_t c = 0; c < Jval.cols; c++) {
						sum += Jval.at(r, c);
					}
				}
				error = sum + regu;

				//layers backward
				for (int k = _networks.size() - 2; k >= 0; k--) {
					Layer<_Tp, _activation>& layer = _networks[k];
					Layer<_Tp, _activation>& layer_next = _networks[k + 1];
					Matrix<_Tp> weights = layer_next.getWeights();
					Matrix<_Tp> err = layer_next.getLossErr();
					layer.backward(weights, err);
				}
				//update weights
				for (int i = _networks.size() - 1; i >= 0; i--) {
					Layer<_Tp, _activation>& layer = _networks[i];
					if (i > 0){
						Layer<_Tp, _activation>& layer_pre = _networks[i - 1];
						layer.update(layer_pre.getActData(), _lr,_lamda);
					} else if (i == 0){
						layer.update(one_batch, _lr, _lamda);
					}
				}
			}
			return error;
		}

		float solve_one_batch(const Matrix<_Tp>& data, const Matrix<_Tp>& rsp){
			float error = FLT_MAX;
			Matrix<_Tp> one_batch = data;
			Matrix<_Tp> one_batch_rsp = rsp;
			Matrix<_Tp> fw_input = one_batch;
			//layers forward 
			for (size_t j = 0; j < _networks.size(); j++) {
				Layer<_Tp, _activation>& layer = _networks[j];
				layer.forward(fw_input, fw_input);
			}

			//calc last layer loss
			Matrix<_Tp> err_L = fw_input - one_batch_rsp;
			_networks[_networks.size() - 1].setLossErr(err_L);

			Matrix<_Tp> Htheta = fw_input;
			Matrix<_Tp> Jval = calc_cost_func_J(Htheta, one_batch_rsp, 1);
			_Tp regu = regulization(_lamda);
			float sum = 0;
			for (size_t r = 0; r < Jval.rows; r++) {
				for (size_t c = 0; c < Jval.cols; c++) {
					sum += Jval.at(r, c);
				}
			}
			error = sum + regu;

			//layers backward
			for (int k = _networks.size() - 2; k >= 0; k--) {
				Layer<_Tp, _activation>& layer = _networks[k];
				Layer<_Tp, _activation>& layer_next = _networks[k + 1];
				Matrix<_Tp> weights = layer_next.getWeights();
				Matrix<_Tp> err = layer_next.getLossErr();
				layer.backward(weights, err);
			}
			//update weights
			for (int i = _networks.size() - 1; i >= 0; i--) {
				Layer<_Tp, _activation>& layer = _networks[i];
				if (i > 0){
					Layer<_Tp, _activation>& layer_pre = _networks[i - 1];
					layer.update(layer_pre.getActData(), _lr, _lamda);
				}
				else if (i == 0){
					layer.update(one_batch, _lr, _lamda);
				}
			}
			return error;
		}

		bool check_net(){
			bool bres = true;
			if (_networks.empty())
				bres = false;
			for (size_t i = 0; i < _networks.size() - 1; i++) {
				Layer<_Tp, _activation>& layer = _networks[i];
				Layer<_Tp, _activation>& layer_next = _networks[i + 1];
				if (layer.nOutputs != layer_next.nInputs)
					bres = false;
			}
			return bres;
		}

		/** @brief calculate cost function $J(\Theta)$
		@param h - input matrix of $h_{\Theta}(x^{i})$
		@param y - input matrix of $y(x^{i})$
		@param method - 
		if method = 0 squared-error cost function,$J(\Theta) = \frac{1}{2m}\sum_{j}(y_{i} - a_{j})^{2}+regularization$
		if method = 1 cross-entropy cost function,$J(\Theta )=-\frac{1}{m}\left [ \sum_{i=1}^{m}\sum_{k=1}^{K}y_{k}^{\left (i\right )}log(h_{\Theta}(x^{(i)}))_{k} + 
		(1-y_{k}^{(i)})log(1-(h_{\Theta}(x^{(i)}))_{k})  \right ]+
		\frac{\lambda }{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_{l}}\sum_{j=1}^{s_{l+1}}(\Theta_{ji}^{l})^{2}$		
		*/
		Matrix<_Tp> calc_cost_func_J(Matrix<_Tp> h, Matrix<_Tp> y,int method = 0){
			assert(h.rows == y.rows);
			assert(h.cols == y.cols);
			Matrix<_Tp> delta;
			Matrix<_Tp> result = Matrix<_Tp>::zeros(1, h.cols);
			switch (method)
			{
			case 0:
				delta = (Matrix<_Tp>)h - (Matrix<_Tp>)y;
				delta = delta.hadamardProduct(delta) / 2;
				for (size_t i = 0; i < delta.rows; i++)	{
					result = result + delta.row(i);
				}
				result = result / delta.rows;
				break;
			case 1:
				delta = Matrix<_Tp>::zeros(h.rows, h.cols);
				for (size_t i = 0; i < h.rows; i++)	{
					for (size_t j = 0; j < h.cols; j++){
						_Tp y_ij = y.at(i, j);
						_Tp h_ij = h.at(i, j);
						delta.at(i, j) = y_ij*log(h_ij) + (1 - y_ij)*log(1 - h_ij);
					}
				}

				for (size_t i = 0; i < delta.rows; i++)	{
					result = result + delta.row(i);
				}
				result = (result / delta.rows) * -1;
				break;
			default:
				break;
			}
			return result;
		}

		_Tp regulization(float lamda){
			float regu = 0;
			for (size_t i = 0; i < _networks.size(); i++) {
				Layer<_Tp, _activation>& layer = _networks[i];
				Matrix<_Tp> weights_and_bias = layer.getWeights();
				Matrix<_Tp> weights = weights_and_bias.subMat(Range_<int>(0, weights_and_bias.rows),
					Range_<int>(0, weights_and_bias.cols - 1));
				weights = weights.hadamardProduct(weights);
				for (size_t m = 0; m < weights.rows; m++) {
					for (size_t n = 0; n < weights.cols; n++){
						regu += weights.at(m, n);
					}
				}
			}
			return (_Tp) lamda * regu / 2.;
		}

	private:
		vector<Layer<_Tp, _activation> > _networks;
		float _epsillon;
		int _maxIter;
		float _lr;
		float _lamda;
	};


	template<typename _Tp>
	void zero_mean(const Matrix<_Tp>& _in, Matrix<_Tp>& _out){
		Matrix<_Tp> in_t = _in;
		in_t = in_t.t();
		_out = Matrix<_Tp>::zeros(in_t.rows, in_t.cols);
		for (size_t i = 0; i < in_t.rows; i++) {
			Matrix<_Tp>& one_row = in_t.row(i);
			//calc mean
			_Tp means = 0;
			_Tp variance = 0;
			for (size_t j = 0; j < one_row.cols; j++) {
				means += one_row.at(0, j);
			}
			means /= one_row.cols;
			//calc variance
			for (size_t j = 0; j < one_row.cols; j++) {
				float v = one_row.at(0, j) - means;
				variance += v * v;
			}
			variance = sqrtf(variance);
			for (size_t j = 0; j < one_row.cols; j++) {
				_out.at(i, j) = (one_row.at(0, j) - means) / variance;
			}
		}
		_out = _out.t();
	}

	template<typename _Tp>
	void normalize(const Matrix<_Tp>& _in, Matrix<_Tp>& _out,double alpha = 0,double beta = 0){
		Matrix<_Tp> in = _in;
		_out = Matrix<_Tp>::zeros(in.rows, in.cols);
		for (size_t i = 0; i < in.rows; i++) {
			Matrix<_Tp>& one_row = in.row(i);
			_Tp minV = alpha, maxV = beta;
			if (alpha == 0 && beta == 0){
				auto pr =
					std::minmax_element(one_row[0].begin(), one_row[0].end());
				minV = *pr.first;	maxV = *pr.second;
			}
			for (size_t j = 0; j < one_row.cols; j++) {
				_out.at(i, j) = (in.at(i, j) - minV) / (maxV - minV);
			}
		}
	}

	void strSplit(std::string& s, std::string delim, std::vector< std::string >& ret)
	{
		size_t last = 0;
		size_t index = s.find_first_of(delim, last);
		while (index != std::string::npos) {
			ret.push_back(s.substr(last, index - last));
			last = index + 1;
			index = s.find_first_of(delim, last);
		}
		if (index - last>0)	{
			ret.push_back(s.substr(last, index - last));
		}
	}

}