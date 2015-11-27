
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<pthread.h>
#include<ctime>
#include<sys/time.h>
#include<assert.h>
using namespace std;

//Ruobing Xie
//Representation Learning of Knowledge Graphs with Entity Descriptions

#define pi 3.1415926535897932384626433832795
#define THREADS_NUM 16

bool L1_flag=1;

//convolutional layer
int window_1 = 2;		//1st window size
int window_2 = 1;		//2nd window size
double **conv_w1;
double **conv_w1_tmp[THREADS_NUM];
double **conv_w2;
double **conv_w2_tmp[THREADS_NUM];
int n_pooling_1 = 4;		//1st layer 4-pooling

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}

double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double res=0;
    for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}

double norm(vector<double> &a)
{
	double x = vec_len(a);
	if (x>1)
	for (int ii=0; ii<a.size(); ii++)
			a[ii]/=x;
	return 0;
}

int rand_max(int x)
{
	int res = (rand()*rand())%x;
	while (res<0)
		res+=x;
	return res;
}

string version;		//bern/unif
char buf[100000],buf1[100000];
int relation_num,entity_num,word_num;
map<string,int> relation2id,entity2id,word2id;
map<int,string> id2entity,id2relation,id2word;
vector<vector<int> > entityWords_vec;
int nbatches, batchsize;

map<int,map<int,int> > left_entity,right_entity;
map<int,double> left_num,right_num;

int n,n_1,n_w,method;		//n£ºdimension of entity/relation, n_1: dimension of feature_vec in 1st-layer CNN, n_w: dimension of word 
double res_triple,res_cnn,res_c_e,res_e_c,res_normal;		//loss function value
double res_thread_triple[THREADS_NUM], res_thread_cnn[THREADS_NUM], res_thread_normal[THREADS_NUM];		//loss for each thread
double res_thread_c_e[THREADS_NUM], res_thread_e_c[THREADS_NUM];
double count,count1;
double rate,rate_c,rate_m,rate_n;		//learning rate
double margin,margin_c,margin_m;		//margin
vector<int> fb_h,fb_l,fb_r;
vector<vector<double> > relation_vec,word_vec,entity_vec;		//embeddings
vector<vector<double> > relation_tmp,word_tmp,entity_tmp;

vector<vector<double> > cnn_vec;		//cnn_vec for entity
vector<vector<double> > cnn_grad;
vector<vector<int> > l2_MaxPooling;
vector<int> cnnVecState;		//entity state

vector<vector<vector<double> > > mid_vec;		//1st-layer output
vector<vector<vector<int> > > l1_MaxPooling;
vector<vector<vector<double> > > mid_grad;

vector<double> posErrorVec[THREADS_NUM];
vector<double> negErrorVec[THREADS_NUM];

pthread_mutex_t mut_mutex;

void sgd();
void train_triple_mul(int, int, int, int, int, int, int);
void train_cnn_mul(int, int, int, int, int, int, int);
void train_e_c_mul(int, int, int, int, int, int, int);
void train_c_e_mul(int, int, int, int, int, int, int);

map<pair<int,int>, map<int,int> > ok;		//mark of positive

void add(int x,int y,int z)
{
	fb_h.push_back(x);
	fb_r.push_back(z);
	fb_l.push_back(y);
	ok[make_pair(x,z)][y]=1;
}

void run(int n_in,int n_1_in,int n_w_in,double rate_in,double margin_in,int method_in)
{
	n = n_in;		//init
	n_1 = n_1_in;
	n_w = n_w_in;
	rate = rate_in;
	rate_c = 1*rate;
	rate_n = 0.1*rate;
	rate_m = 0.4*rate;
	margin = margin_in;
	margin_c = 1*margin;
	margin_m = 0.8*margin;
	method = method_in;
	
	relation_vec.resize(relation_num);
	for (int i=0; i<relation_vec.size(); i++)
		relation_vec[i].resize(n);
	entity_vec.resize(entity_num);
	for (int i=0; i<entity_vec.size(); i++)
		entity_vec[i].resize(n);
	relation_tmp.resize(relation_num);
	for (int i=0; i<relation_tmp.size(); i++)
		relation_tmp[i].resize(n);
	entity_tmp.resize(entity_num);
	for (int i=0; i<entity_tmp.size(); i++)
		entity_tmp[i].resize(n);
	for (int i=0; i<relation_num; i++)		//init
	{
		for (int ii=0; ii<n; ii++)
			relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));		//pre-trained embeddings are optional
	}
	for (int i=0; i<entity_num; i++)
	{
		for (int ii=0; ii<n; ii++)
			entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));		//pre-trained embeddings are optional
		norm(entity_vec[i]);
	}
	
	//init by pre-trained word embeddings
	word_vec.resize(word_num);
	for (int i=0; i<word_vec.size(); i++)
		word_vec[i].resize(n_w);
	for (int i=0; i<word_num; i++)
	{
		for (int ii=0; ii<n_w; ii++)
			word_vec[i][ii] = randn(0,1.0/n_w,-6/sqrt(n_w),6/sqrt(n_w));
		norm(word_vec[i]);
	}
	stringstream ss;
	ss << n_w; 
	string n_w_string = "../../../word2vec-41/word_vector_" + ss.str() + ".txt";
	FILE* f1 = fopen(n_w_string.c_str(), "r");
	while (fscanf(f1,"%s",buf)==1)
	{
		string st=buf;
		if(word2id.count(st)==0)
			continue;
		int tempWordID = word2id[st];
		for (int ii=0; ii<n_w; ii++)
			fscanf(f1,"%lf",&word_vec[tempWordID][ii]);
		norm(word_vec[tempWordID]);
	}
	
	//convolution init
	conv_w1 = new double *[n_1];
	for (int i=0; i<n_1; i++)
	{
		conv_w1[i] = new double[n_w*window_1];
		for (int ii=0; ii<n_w*window_1; ii++)
		{
			conv_w1[i][ii] = randn(0,1.0/(n_w*window_1),-6/sqrt(n_w*window_1),6/sqrt(n_w*window_1));
		}
	}
	conv_w2 = new double *[n];
	for (int i=0; i<n; i++)
	{
		conv_w2[i] = new double[n_1*window_2];
		for (int ii=0; ii<n_1*window_2; ii++)
		{
			conv_w2[i][ii] = randn(0,1.0/(n_1*window_2),-6/sqrt(n_1*window_2),6/sqrt(n_1*window_2));
		}
	}
	
	for(int k=0; k<THREADS_NUM; k++)
	{
		conv_w1_tmp[k] = new double *[n_1];
		for (int i=0; i<n_1; i++)
		{
			conv_w1_tmp[k][i] = new double[n_w*window_1];
			for (int ii=0; ii<n_w*window_1; ii++)
			{
				conv_w1_tmp[k][i][ii] = 0;
			}
		}
	}
	for(int k = 0;k<THREADS_NUM;k++)
	{
		conv_w2_tmp[k] = new double *[n];
		for (int i=0; i<n; i++)
		{
			conv_w2_tmp[k][i] = new double[n_1*window_2];
			for (int ii=0; ii<n_1*window_2; ii++)
			{
				conv_w2_tmp[k][i][ii] = 0;
			}
		}
	}
	
	//cnn vec init
	cnn_vec.resize(entity_num);
	for (int i=0; i<cnn_vec.size(); i++)
		cnn_vec[i].resize(n);
	cnn_grad.resize(entity_num);
	for (int i=0; i<cnn_grad.size(); i++)
	{
		cnn_grad[i].resize(n);
		for(int j=0; j<n; j++)
		{
			cnn_grad[i][j] = 0;
		}
	}
	l2_MaxPooling.resize(entity_num);
	for (int i=0; i<l2_MaxPooling.size(); i++)
		l2_MaxPooling[i].resize(n);
	mid_vec.resize(entity_num);
	for (int i=0; i<mid_vec.size(); i++)
	{
		int l1_length = (entityWords_vec[i].size()-1)/n_pooling_1+1;
		mid_vec[i].resize(l1_length);
		for(int j = 0;j<l1_length;j++)
			mid_vec[i][j].resize(n_1);
	}
	l1_MaxPooling.resize(entity_num);
	for (int i=0; i<l1_MaxPooling.size(); i++)
	{
		int l1_length = (entityWords_vec[i].size()-1)/n_pooling_1+1;
		l1_MaxPooling[i].resize(l1_length);
		for(int j = 0;j<l1_length;j++)
			l1_MaxPooling[i][j].resize(n_1);
	}
	mid_grad.resize(entity_num);
	for (int i=0; i<mid_grad.size(); i++)
	{
		int l1_length = (entityWords_vec[i].size()-1)/n_pooling_1+1;
		mid_grad[i].resize(l1_length);
		for(int j = 0;j<l1_length;j++)
			mid_grad[i][j].resize(n_1);
	}
	
	cnnVecState.resize(entity_num);
	for(int k = 0;k<THREADS_NUM;k++)
	{
		posErrorVec[k].resize(n);
		negErrorVec[k].resize(n);
	}
	
	mut_mutex = PTHREAD_MUTEX_INITIALIZER;
	sgd();
}

void *rand_sel(void *tid_void)		//multi-thread train
{
	long tid = (long) tid_void;
	for (int k=0; k<batchsize; k++)
	{
		int i=rand_max(fb_h.size());
		int j=rand_max(entity_num);
		double pr = 1000*right_num[fb_r[i]]/(right_num[fb_r[i]]+left_num[fb_r[i]]);
		if (method ==0)
			pr = 500;
		
		//||entity_h+r-entity_t||
		int flag_num = rand_max(1000);
		if (flag_num<pr)
		{
			while (ok.count(make_pair(fb_h[i],fb_r[i]))>0&&ok[make_pair(fb_h[i],fb_r[i])].count(j)>0)
				j=rand_max(entity_num);
			train_triple_mul(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		}
		else
		{
			while (ok.count(make_pair(j,fb_r[i]))>0&&ok[make_pair(j,fb_r[i])].count(fb_l[i])>0)
				j=rand_max(entity_num);
			train_triple_mul(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		}
		int rel_neg = rand_max(relation_num);		//negative relation
		while (ok.count(make_pair(fb_h[i], rel_neg))>0&& ok[make_pair(fb_h[i], rel_neg)].count(fb_l[i]) > 0)
			rel_neg = rand_max(relation_num);
		train_triple_mul(fb_h[i],fb_l[i],fb_r[i],fb_h[i],fb_l[i],rel_neg,tid);
		
		//||cnn_h+r-cnn_t||
		if (flag_num<pr)
			train_cnn_mul(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		else
			train_cnn_mul(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		train_cnn_mul(fb_h[i], fb_l[i], fb_r[i], fb_h[i], fb_l[i],rel_neg,tid);
		
		//||entity_h+r-cnn_t||
		if (flag_num<pr)
			train_e_c_mul(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		else
			train_e_c_mul(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		train_e_c_mul(fb_h[i], fb_l[i], fb_r[i], fb_h[i], fb_l[i],rel_neg,tid);
		
		//||cnn_h+r-entity_t||
		if (flag_num<pr)
			train_c_e_mul(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		else
			train_c_e_mul(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		train_c_e_mul(fb_h[i], fb_l[i], fb_r[i], fb_h[i], fb_l[i],rel_neg,tid);
		
		//normalization
		norm(relation_tmp[fb_r[i]]);
		norm(relation_tmp[rel_neg]);
		norm(entity_tmp[fb_h[i]]);
		norm(entity_tmp[fb_l[i]]);
		norm(entity_tmp[j]);
	}
}

void *update_grad(void *tid_void)		//multi-thread update
{
	long tid = (long) tid_void;
	int block_size = entity_num / THREADS_NUM;
	int begin_e = tid * block_size;
	int end_e = (tid+1) * block_size;
	if(tid == THREADS_NUM-1)
		end_e = entity_num;
	//updata conv & word_vec
	double nonLinear_1 = 0, der_1 = 0, der_2 = 0;
	for(int ent=begin_e; ent<end_e; ent++)
	{
		if(cnnVecState[ent] != 2)		//no need to update
			continue;
		//init
		for(int i=0; i<mid_grad[ent].size(); i++)
			for(int j=0; j<n_1; j++)
				mid_grad[ent][i][j] = 0;
		for(int i=0; i<n; i++)
		{
			der_2 = cnn_grad[ent][i]/mid_vec[ent].size();
			cnn_grad[ent][i] = 0;
			for(int h = 0;h<mid_vec[ent].size();h++)
			{
				for(int ii = 0;ii<window_2;ii++)
				{
					int tempMidID = h+ii;
					if(tempMidID >= mid_vec[ent].size())		//zero-padding
						break;
					for(int k = 0;k<n_1;k++)
					{
						conv_w2_tmp[tid][i][ii*n_1+k] += der_2*mid_vec[ent][tempMidID][k];
						mid_grad[ent][tempMidID][k] += der_2*conv_w2[i][ii*n_1+k];
					}
				}
			}
		}
		for(int tempMidID=0; tempMidID<mid_grad[ent].size(); tempMidID++)
		{
			for(int k=0; k<n_1; k++)
			{
				nonLinear_1 = 1-mid_vec[ent][tempMidID][k]*mid_vec[ent][tempMidID][k];
				der_1 = mid_grad[ent][tempMidID][k]*nonLinear_1;
				for(int iii = 0;iii<window_1;iii++)
				{
					if(l1_MaxPooling[ent][tempMidID][k]+iii >=  entityWords_vec[ent].size())
						break;
					int tempWordID = entityWords_vec[ent][l1_MaxPooling[ent][tempMidID][k]+iii];
					for(int j = 0;j<n_w;j++)
					{
						conv_w1_tmp[tid][k][iii*n_w+j] += der_1*word_vec[tempWordID][j];
						word_tmp[tempWordID][j] += der_1*conv_w1[k][iii*n_w+j];
					}
				}
			}
		}
		for(int ii=0;ii<entityWords_vec[ent].size();ii++)
		{
			norm(word_tmp[entityWords_vec[ent][ii]]);
		}
	}
}

void update_multithread()		//update others
{
	//update
	relation_vec = relation_tmp;
	entity_vec = entity_tmp;
	word_vec = word_tmp;
	for(int k = 0;k<THREADS_NUM;k++)
	{
		for (int i=0; i<n_1; i++)
		{
			for (int ii=0; ii<n_w*window_1; ii++)
				conv_w1[i][ii] += conv_w1_tmp[k][i][ii];
		}
		for (int i=0; i<n; i++)
		{
			for (int ii=0; ii<n_1*window_2; ii++)
				conv_w2[i][ii] += conv_w2_tmp[k][i][ii];
		}
		res_triple += res_thread_triple[k];
		res_cnn += res_thread_cnn[k];
		res_c_e += res_thread_c_e[k];
		res_e_c += res_thread_e_c[k];
		res_normal += res_thread_normal[k];
	}
}

void sgd()		//SGD training
{
	res_triple=0;
	res_cnn = 0;
	res_c_e = 0;
	res_e_c = 0;
	res_normal = 0;
	nbatches=100;		//block number
	int nepoch = 1000;		//iter
	batchsize = fb_h.size()/nbatches/THREADS_NUM;		//mini_batch size for each thread
	cout << "batchsize : " << batchsize << endl;
	for (int epoch=0; epoch<nepoch; epoch++)
	{
		res_triple=0;
		res_cnn = 0;
		res_c_e = 0;
		res_e_c = 0;
		res_normal = 0;
		for (int batch = 0; batch<nbatches; batch++)
		{
			for(int i = 0;i<entity_num;i++)
				cnnVecState[i] = 0;
			for(int k = 0;k<THREADS_NUM;k++)		//init
			{
				for (int i=0; i<n_1; i++)
					for (int ii=0; ii<n_w*window_1; ii++)
						conv_w1_tmp[k][i][ii] = 0;
				for (int i=0; i<n; i++)
					for (int ii=0; ii<n_1*window_2; ii++)
						conv_w2_tmp[k][i][ii] = 0;
				res_thread_triple[k] = 0;
				res_thread_cnn[k] = 0;
				res_thread_c_e[k] = 0;
				res_thread_e_c[k] = 0;
				res_thread_normal[k] = 0;
			}
			relation_tmp = relation_vec;
			entity_tmp = entity_vec;
			word_tmp = word_vec;
			//multi-thread for train
			pthread_t threads[THREADS_NUM];
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_create(&threads[k], NULL, rand_sel, (void *)k);		//train
			}
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_join(threads[k], NULL);
			}
			//multi-thread for update
			pthread_t threads2[THREADS_NUM];
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_create(&threads2[k], NULL, update_grad, (void *)k);		//update
			}
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_join(threads2[k], NULL);
			}
			//update other
			update_multithread();
			//cout << "update once : " << batch << endl;
		}
		cout<<"epoch:"<<epoch<<' '<<res_triple<< ' ' << res_cnn << ' ';
		cout << res_c_e << ' ' << res_e_c << ' ' << res_normal << endl;
		//output
		FILE* f2 = fopen(("../res/relation2vec."+version).c_str(),"w");
		FILE* f3 = fopen(("../res/entity2vec."+version).c_str(),"w");
		FILE* f4 = fopen(("../res/word2vec."+version).c_str(),"w");
		FILE* f5 = fopen(("../res/weight1."+version).c_str(),"w");
		FILE* f6 = fopen(("../res/weight2."+version).c_str(),"w");
		for (int i=0; i<relation_num; i++)
		{
			for (int ii=0; ii<n; ii++)
				fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
			fprintf(f2,"\n");
		}
		for (int i=0; i<entity_num; i++)
		{
			for (int ii=0; ii<n; ii++)
				fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
			fprintf(f3,"\n");
		}
		for (int i=0; i<word_num; i++)
		{
			for (int ii=0; ii<n_w; ii++)
				fprintf(f4,"%.6lf\t",word_vec[i][ii]);
			fprintf(f4,"\n");
		}
		for (int i=0; i<n_1; i++)
		{
			for (int ii=0; ii<n_w*window_1; ii++)
				fprintf(f5,"%.6lf\t",conv_w1[i][ii]);
			fprintf(f5,"\n");
		}
		for (int i=0; i<n; i++)
		{
			for (int ii=0; ii<n_1*window_2; ii++)
				fprintf(f6,"%.6lf\t",conv_w2[i][ii]);
			fprintf(f6,"\n");
		}
		fclose(f2);
		fclose(f3);
		fclose(f4);
		fclose(f5);
		fclose(f6);
	}
}

//for cnn_vec length
void gradient_normalization(int e1,int tid)
{
	for (int i=0; i<n; i++)
	{
		double nonLinear_2 = 1-cnn_vec[e1][i]*cnn_vec[e1][i];
		cnn_grad[e1][i] -= rate_n*2*cnn_vec[e1][i]*nonLinear_2;
	}
}

//calc cnn_vec
void calc_cnn_vec(int ent)		//use CNN to get entity description-based embedding
{
	//1st convolutional layer & max-pooling layer
	int l1_length = (entityWords_vec[ent].size()-1)/n_pooling_1+1;
	for(int k = 0;k<l1_length;k++)
	{
		for(int j = 0;j<n_1;j++)
		{
			double tempMax = -2147483647;		//small number
			l1_MaxPooling[ent][k][j] = -1;
			for(int i = n_pooling_1*k;i<n_pooling_1*(k+1);i++)
			{
				if(i >= entityWords_vec[ent].size())
					break;
				double tempTokenValue = 0;
				for(int ii = 0;ii<window_1;ii++)
				{
					if(i+ii >= entityWords_vec[ent].size())
						break;
					int tempWordID = entityWords_vec[ent][i+ii];
					for(int iii = 0;iii<n_w;iii++)
					{
						tempTokenValue += word_vec[tempWordID][iii] * conv_w1[j][ii*n_w+iii];
					}
				}
				if(tempMax < tempTokenValue)
				{
					tempMax = tempTokenValue;
					l1_MaxPooling[ent][k][j] = i;
				}
			}
			//hyperpolic tangent
			double tempExpo = exp(-2*tempMax);
			mid_vec[ent][k][j] = (1-tempExpo) / (1+tempExpo);
		}
	}
	//2nd convolutional layer & max-pooling layer
	for(int j = 0;j<n;j++)
	{
		double tempExpo = 0;
		int tempWordNum = 0;
		for(int i = 0;i<l1_length;i++)
		{
			double tempTokenValue = 0;
			for(int ii = 0;ii<window_2;ii++)
			{
				if(i+ii >= l1_length)
					break;
				for(int iii = 0;iii<n_1;iii++)
				{
					tempTokenValue += mid_vec[ent][i+ii][iii] * conv_w2[j][ii*n_1+iii];
				}
			}
			//mean pooling
			tempExpo += tempTokenValue;
			tempWordNum++;
		}
		//hyperpolic tangent
		tempExpo /= tempWordNum;
		tempExpo = exp(-2*tempExpo);
		cnn_vec[ent][j] = (1-tempExpo) / (1+tempExpo);
	}
}

double calc_sum_cnn(int e1,int e2,int rel, int flag, int tid)		//similarity
{
	double sum=0;
	//CNN part
	if(cnnVecState[e1] == 0)
	{
		calc_cnn_vec(e1);
		double sum1 = vec_len(cnn_vec[e1]);
		if(sum1 > 1)
		{
			res_thread_normal[tid] += sum1-1;
			gradient_normalization(e1,tid);
		}
		cnnVecState[e1] = 1;
	}
	if(cnnVecState[e2] == 0)
	{
		calc_cnn_vec(e2);
		double sum1 = vec_len(cnn_vec[e2]);
		if(sum1 > 1)
		{
			res_thread_normal[tid] += sum1-1;
			gradient_normalization(e2,tid);
		}
		cnnVecState[e2] = 1;
	}
	if(flag == 1)		//positive_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					posErrorVec[tid][ii] = 1;
				else
					posErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				posErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
	else		//negative_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					negErrorVec[tid][ii] = 1;
				else
					negErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				negErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
}

double calc_sum_c_e(int e1,int e2,int rel, int flag, int tid)
{
	double sum=0;
	if(flag == 1)		//positive_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					posErrorVec[tid][ii] = 1;
				else
					posErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				posErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
	else		//negative_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					negErrorVec[tid][ii] = 1;
				else
					negErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				negErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
}

double calc_sum_e_c(int e1,int e2,int rel, int flag, int tid)
{
	double sum=0;
	if(flag == 1)		//positive_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					posErrorVec[tid][ii] = 1;
				else
					posErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				posErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
	else		//negative_sign
	{
		if (L1_flag)		//L1
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)
					negErrorVec[tid][ii] = 1;
				else
					negErrorVec[tid][ii] = -1;
			}
		}
		else		//L2
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = cnn_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				negErrorVec[tid][ii] = 2*tempSum;
			}
		}
		return sum;
	}
}

void gradient_cnn(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//SGD update
{
	double nonLinear_2 = 0;
	//update state
	cnnVecState[e1_a] = 2;
	cnnVecState[e2_a] = 2;
	cnnVecState[e1_b] = 2;
	cnnVecState[e2_b] = 2;
	for (int i = 0;i<n;i++)
	{
		//relation
		relation_tmp[rel_a][i] += rate_c*posErrorVec[tid][i];
		//head
		nonLinear_2 = 1-cnn_vec[e1_a][i]*cnn_vec[e1_a][i];
		cnn_grad[e1_a][i] += rate_c*posErrorVec[tid][i]*nonLinear_2;
		//tail
		nonLinear_2 = 1-cnn_vec[e2_a][i]*cnn_vec[e2_a][i];
		cnn_grad[e2_a][i] -= rate_c*posErrorVec[tid][i]*nonLinear_2;
		//relation
		relation_tmp[rel_b][i] -= rate_c*negErrorVec[tid][i];
		//head
		nonLinear_2 = 1-cnn_vec[e1_b][i]*cnn_vec[e1_b][i];
		cnn_grad[e1_b][i] -= rate_c*negErrorVec[tid][i]*nonLinear_2;
		//tail
		nonLinear_2 = 1-cnn_vec[e2_b][i]*cnn_vec[e2_b][i];
		cnn_grad[e2_b][i] += rate_c*negErrorVec[tid][i]*nonLinear_2;
	}
}

void gradient_c_e(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//SGD update
{
	double nonLinear_2 = 0;
	//update state
	cnnVecState[e1_a] = 2;
	cnnVecState[e2_a] = 2;
	cnnVecState[e1_b] = 2;
	cnnVecState[e2_b] = 2;
	for (int i = 0;i<n;i++)
	{
		//relation
		relation_tmp[rel_a][i] += rate_m*posErrorVec[tid][i];
		//head
		nonLinear_2 = 1-cnn_vec[e1_a][i]*cnn_vec[e1_a][i];
		cnn_grad[e1_a][i] += rate_m*posErrorVec[tid][i]*nonLinear_2;
		//tail
		entity_tmp[e2_a][i] -= rate_m*posErrorVec[tid][i];
		//relation
		relation_tmp[rel_b][i] -= rate_m*negErrorVec[tid][i];
		//head
		nonLinear_2 = 1-cnn_vec[e1_b][i]*cnn_vec[e1_b][i];
		cnn_grad[e1_b][i] -= rate_m*negErrorVec[tid][i]*nonLinear_2;
		//tail
		entity_tmp[e2_b][i] += rate_m*negErrorVec[tid][i];
	}
}

void gradient_e_c(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//SGD update
{
	double nonLinear_2 = 0;
	//update state
	cnnVecState[e1_a] = 2;
	cnnVecState[e2_a] = 2;
	cnnVecState[e1_b] = 2;
	cnnVecState[e2_b] = 2;
	for (int i = 0;i<n;i++)
	{
		//relation
		relation_tmp[rel_a][i] += rate_m*posErrorVec[tid][i];
		//head
		entity_tmp[e1_a][i] += rate_m*posErrorVec[tid][i];
		//tail
		nonLinear_2 = 1-cnn_vec[e2_a][i]*cnn_vec[e2_a][i];
		cnn_grad[e2_a][i] -= rate_m*posErrorVec[tid][i]*nonLinear_2;
		//relation
		relation_tmp[rel_b][i] -= rate_m*negErrorVec[tid][i];
		//head
		entity_tmp[e1_b][i] -= rate_m*negErrorVec[tid][i];
		//tail
		nonLinear_2 = 1-cnn_vec[e2_b][i]*cnn_vec[e2_b][i];
		cnn_grad[e2_b][i] += rate_m*negErrorVec[tid][i]*nonLinear_2;
	}
}

void train_cnn_mul(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//||entity_vec-cnn_vec||
{
	double sum1 = calc_sum_cnn(e1_a,e2_a,rel_a,1,tid);
	double sum2 = calc_sum_cnn(e1_b,e2_b,rel_b,0,tid);
	if (sum1+margin_c>sum2)
	{
		res_thread_cnn[tid]+=margin_c+sum1-sum2;
		gradient_cnn( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

void train_c_e_mul(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//||cnn_h+r-entity_t||
{
	double sum1 = calc_sum_c_e(e1_a,e2_a,rel_a,1,tid);
	double sum2 = calc_sum_c_e(e1_b,e2_b,rel_b,0,tid);
	if (sum1+margin_m>sum2)
	{
		res_thread_c_e[tid]+=margin_m+sum1-sum2;
		gradient_c_e( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

void train_e_c_mul(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//||entity_h+r-cnn_t||
{
	double sum1 = calc_sum_e_c(e1_a,e2_a,rel_a,1,tid);
	double sum2 = calc_sum_e_c(e1_b,e2_b,rel_b,0,tid);
	if (sum1+margin_m>sum2)
	{
		res_thread_e_c[tid]+=margin_m+sum1-sum2;
		gradient_e_c( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

double calc_triple_sum(int e1,int e2,int rel)		//for ||h+r-t||
{
	double sum=0;
	if (L1_flag)
		for (int ii=0; ii<n; ii++)
			sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
	else
		for (int ii=0; ii<n; ii++)
			sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
	return sum;
}

void gradient_triple(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)		//SGD update
{
	for (int ii=0; ii<n; ii++)
	{
		double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
		if (L1_flag)
			if (x>0)
				x=1;
			else
				x=-1;
		relation_tmp[rel_a][ii]-=-1*rate*x;
		entity_tmp[e1_a][ii]-=-1*rate*x;
		entity_tmp[e2_a][ii]+=-1*rate*x;
		
		x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);
		if (L1_flag)
			if (x>0)
				x=1;
			else
				x=-1;
		relation_tmp[rel_b][ii]-=rate*x;
		entity_tmp[e1_b][ii]-=rate*x;
		entity_tmp[e2_b][ii]+=rate*x;
	}
}

void train_triple_mul(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)
{
	double sum1 = calc_triple_sum(e1_a,e2_a,rel_a);
	double sum2 = calc_triple_sum(e1_b,e2_b,rel_b);
	if (sum1+margin>sum2)
	{
		res_thread_triple[tid]+=margin+sum1-sum2;
		gradient_triple( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

void prepare()
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	FILE* f3 = fopen("../data/word2id.txt","r");
	FILE* f4 = fopen("../data/entityWords.txt","r");
	int x;
	//build entity2ID¡¢ID2entity map
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;		//<entity,ID>
		id2entity[x]=st;		//<ID,entity>
		entity_num++;
	}
	//build relation2ID¡¢ID2relation map
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
	//build word2ID¡¢ID2word map
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		word2id[st]=x;		//<word,ID>
		id2word[x]=st;		//<ID,word>
		word_num++;
	}
	//build entityWords_vec 
	entityWords_vec.resize(entity_num);
	while (fscanf(f4,"%s%d",buf,&x)==2)
	{
		string st=buf;		//entity
		int temp_entity_id = entity2id[st];
		int temp_word_num = x;
		entityWords_vec[temp_entity_id].resize(temp_word_num);
		for(int i = 0;i<temp_word_num;i++)
		{
			fscanf(f4,"%s",buf);
			string st1=buf;		//word
			entityWords_vec[temp_entity_id][i] = word2id[st1];
		}
	}
	//read triple set
    FILE* f_kb = fopen("../data/train.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;		//head
        fscanf(f_kb,"%s",buf);
        string s2=buf;		//tail
        fscanf(f_kb,"%s",buf);
        string s3=buf;		//relation
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        left_entity[relation2id[s3]][entity2id[s1]]++;
        right_entity[relation2id[s3]][entity2id[s2]]++;
        add(entity2id[s1],entity2id[s2],relation2id[s3]);
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	left_num[i]=sum2/sum1;
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	right_num[i]=sum2/sum1;
    }
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int method = 1;
    int n = 50;
	int n_1 = 50;
	int n_w = 50;
    double rate = 0.001;		//learning rate
    double margin = 1;		//loss margin
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size1", argc, argv)) > 0) n_1 = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sizew", argc, argv)) > 0) n_w = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
	cout<<"size of layer1 = "<<n_1<<endl;
	cout<<"size of word = "<<n_w<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
    if (method)
        version = "bern";
    else
        version = "unif";
    cout<<"method = "<<version<<endl;
    prepare();
    run(n,n_1,n_w,rate,margin,method);
}