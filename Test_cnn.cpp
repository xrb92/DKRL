#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>

#include <fstream>

using namespace std;



bool debug=false;
bool L1_flag=1;

//convolutional layer weight
int window_1 = 2;		//1st window size
int window_2 = 1;		//2nd window size
double **conv_w1;		//1st convolutional layer weight matrix, n1*(n_w*window)
double **conv_w2;		//1st convolutional layer weight matrix, n*(n_1*window)
int n_pooling_1 = 4;		//1st layer 4-pooling

string version;
string trainortest = "test";

map<int,string> id2entity,id2relation,id2word;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int> e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;

int relation_num,entity_num,word_num;		//记录relation与entity的数目
map<string,int> relation2id,entity2id,word2id;		//<relation,ID>
vector<vector<int> > entityWords_vec;		//记录各entity的特征词word_ID

int n = 100;
int n_1 = 100;
int n_w = 100;		//dimention of words

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%9==4)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,cnn_vec,word_vec;


    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    void add(int x,int y,int z, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }
	
	double norm(vector<double> &a)		//向量正则化
    {
        double x = vec_len(a);		//返回a的模长
        if (x>1)		//只有模长大于1时需要正则化
        for (int ii=0; ii<a.size(); ii++)
                a[ii]/=x;
        return 0;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
        	for (int ii=0; ii<n; ii++)
            sum+=-fabs(cnn_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii]);
        else
        for (int ii=0; ii<n; ii++)
            sum+=-sqr(cnn_vec[e2][ii]-cnn_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    }
    void run()
    {
		//relation_vec
        FILE* f1 = fopen(("../res/relation2vec."+version).c_str(),"r");		//训练模型文件
        cout<<relation_num<<' '<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
		//第一层卷积层weight matrix
		FILE* f2 = fopen(("../res/weight1."+version).c_str(),"r");
		conv_w1 = new double *[n_1];
		for (int i=0; i<n_1; i++)		//随机赋初值
        {
			conv_w1[i] = new double[n_w*window_1];
            for (int ii=0; ii<n_w*window_1; ii++)
			{
				fscanf(f2,"%lf",&conv_w1[i][ii]);
			}
        }
		//第二层卷积层weight matrix
		FILE* f3 = fopen(("../res/weight2."+version).c_str(),"r");
		conv_w2 = new double *[n];
		for (int i=0; i<n; i++)		//随机赋初值
        {
			conv_w2[i] = new double[n_1*window_2];
            for (int ii=0; ii<n_1*window_2; ii++)
			{
				fscanf(f3,"%lf",&conv_w2[i][ii]);
			}
        }
		//word_vec
		FILE* f4 = fopen(("../res/word2vec."+version).c_str(),"r");
		word_vec.resize(word_num);		//word_vec分配空间
		for (int i=0; i<word_vec.size(); i++)
			word_vec[i].resize(n_w);
		for (int i=0; i<word_num;i++)
        {
            for (int ii=0; ii<n_w; ii++)
                fscanf(f4,"%lf",&word_vec[i][ii]);
            if (vec_len(word_vec[i])-1>1e-3)
            	cout<<"wrong_word"<<i<<' '<<vec_len(word_vec[i])<<endl;
        }
        fclose(f1);
		fclose(f2);
        fclose(f3);
		fclose(f4);
		
		//构建cnn_vec
		cnn_vec.resize(entity_num);
		for(int ent=0;ent<entity_num;ent++)
		{
			//计算第一层convolutional layer & max-pooling layer
			cnn_vec[ent].resize(n);
			int l1_length = (entityWords_vec[ent].size()-1)/n_pooling_1+1;		//4-pooling
			vector<vector<double> > mid_vec;
			mid_vec.resize(l1_length);		//init
			for(int k=0;k<l1_length;k++)
				mid_vec[k].resize(n_1);
			for(int k = 0;k<l1_length;k++)		//对第一层每个block计算结果并取max
			{
				for(int j = 0;j<n_1;j++)		//对于n个维度
				{
					double tempMax = -2147483640;		//-INT_MAX
					for(int i = n_pooling_1*k;i<n_pooling_1*(k+1);i++)		//对于entity words window
					{
						if(i >= entityWords_vec[ent].size())		//add all-zero padding
							break;
						double tempTokenValue = 0;
						for(int ii = 0;ii<window_1;ii++)		//window
						{
							if(i+ii >= entityWords_vec[ent].size())		//add all-zero padding
								break;
							int tempWordID = entityWords_vec[ent][i+ii];		//当前的wordid
							for(int iii = 0;iii<n_w;iii++)		//对于word embedding的每一维
							{
								tempTokenValue += word_vec[tempWordID][iii] * conv_w1[j][ii*n_w+iii];
							}
						}
						if(tempMax < tempTokenValue)
							tempMax = tempTokenValue;
					}
					//加入非线性层hyperpolic tangent
					double tempExpo = exp(-2*tempMax);
					mid_vec[k][j] = (1-tempExpo) / (1+tempExpo);		//记录entity embedding
				}
			}
			//计算第二层convolutional layer & max-pooling layer
			for(int j = 0;j<n;j++)		//对于n个维度
			{
				double tempExpo = 0;
				int tempWordNum = 0;
				for(int i = 0;i<l1_length;i++)		//对于entity words window
				{
					double tempTokenValue = 0;
					for(int ii = 0;ii<window_2;ii++)		//window
					{
						if(i+ii >= l1_length)		//add all-zero padding
							break;
						for(int iii = 0;iii<n_1;iii++)		//对于mid_vec的每一维
						{
							tempTokenValue += mid_vec[i+ii][iii] * conv_w2[j][ii*n_1+iii];
						}
					}
					//mean pooling
					tempExpo += tempTokenValue;
					tempWordNum++;
				}
				//加入非线性层hyperpolic tangent
				tempExpo /= tempWordNum;
				tempExpo = exp(-2*tempExpo);
				cnn_vec[ent][j] = (1-tempExpo) / (1+tempExpo);		//记录entity embedding
			}
			if(vec_len(cnn_vec[ent]) > 1)
				cout << ent << ' ' << vec_len(cnn_vec[ent]) << endl;
		}
		
		//test
		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double mid_sum = 0,mid_sum_filter=0;
		double lp_n=0,lp_n_filter = 0;
		double rp_n=0,rp_n_filter = 0;
		double mid_p_n=0,mid_p_n_filter = 0;
		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> mid_sum_r,mid_sum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,double> mid_p_n_r,mid_p_n_filter_r;
		map<int,int> rel_num;

		int hit_relation = 1;		//hits n，top n中正确的数
		int hit_entity = 10;		//hits n，top n中正确的数
		for (int testid = 0; testid<fb_l.size(); testid+=1)		//循环
		{
			int h = fb_h[testid];
			int l = fb_l[testid];
			int rel = fb_r[testid];
			double tmp = calc_sum(h,l,rel);
			rel_num[rel]+=1;
			vector<pair<int,double> > a;
			for (int i=0; i<entity_num; i++)		//针对head的预测，对所有entity计算得分
			{
				double sum = calc_sum(i,l,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			double ttt=0;
			int filter = 0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(a[i].first,rel)].count(l)>0)
					ttt++;
			    if (ok[make_pair(a[i].first,rel)].count(l)==0)
			    	filter+=1;
				if (a[i].first ==h)
				{
					lsum+=a.size()-i;
					lsum_filter+=filter+1;
					lsum_r[rel]+=a.size()-i;
					lsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_entity)
					{
						lp_n+=1;
						lp_n_r[rel]+=1;
					}
					if (filter<hit_entity)
					{
						lp_n_filter+=1;
						lp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			a.clear();
			for (int i=0; i<entity_num; i++)		//针对tail的预测，对所有entity计算得分
			{
				double sum = calc_sum(h,i,rel);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{

				if (ok[make_pair(h,rel)].count(a[i].first)>0)
					ttt++;
				if (ok[make_pair(h,rel)].count(a[i].first)==0)
			    	filter+=1;
				if (a[i].first==l)
				{
					rsum+=a.size()-i;
					rsum_filter+=filter+1;
					rsum_r[rel]+=a.size()-i;
					rsum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_entity)
					{
						rp_n+=1;
						rp_n_r[rel]+=1;
					}
					if (filter<hit_entity)
					{
						rp_n_filter+=1;
						rp_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			a.clear();
			for (int i=0; i<relation_num; i++)		//针对relation的预测，对所有relation计算得分
			{
				double sum = calc_sum(h,l,i);
				a.push_back(make_pair(i,sum));
			}
			sort(a.begin(),a.end(),cmp);
			ttt=0;
			filter=0;
			for (int i=a.size()-1; i>=0; i--)
			{
				if (ok[make_pair(h,a[i].first)].count(l)>0)		//ttt为在test中存在此relation
					ttt++;
				if (ok[make_pair(h,a[i].first)].count(l)==0)		//filter为在test中存在此relation
			    	filter+=1;
				if (a[i].first==rel)
				{
					mid_sum+=a.size()-i;		//记录结果rank
					mid_sum_filter+=filter+1;
					mid_sum_r[rel]+=a.size()-i;
					mid_sum_filter_r[rel]+=filter+1;
					if (a.size()-i<=hit_relation)		//统计hits n
					{
						mid_p_n+=1;
						mid_p_n_r[rel]+=1;
					}
					if (filter<hit_relation)
					{
						mid_p_n_filter+=1;
						mid_p_n_filter_r[rel]+=1;
					}
					break;
				}
			}
			if (testid%100==0)
			{
				cout<<testid<<":"<<"\t"<<lsum/(testid+1)<<' '<<lp_n/(testid+1)<<' '<<rsum/(testid+1)<<' '<<rp_n/(testid+1)<<"\t"<<lsum_filter/(testid+1)<<' '<<lp_n_filter/(testid+1)<<' '<<rsum_filter/(testid+1)<<' '<<rp_n_filter/(testid+1)<<endl;
				cout<<"mid:\t"<<mid_sum/(testid+1)<<' '<<mid_p_n/(testid+1)<<"\t"<<mid_sum_filter/(testid+1)<<' '<<mid_p_n_filter/(testid+1)<<endl;
			}
		}
		//输出到文件
		ofstream fout;
		fout.open("../res.txt");
		fout<<"left:"<<lsum/fb_l.size()<<'\t'<<lp_n/fb_l.size()<<"\t"<<lsum_filter/fb_l.size()<<'\t'<<lp_n_filter/fb_l.size()<<endl;
		fout<<"right:"<<rsum/fb_r.size()<<'\t'<<rp_n/fb_r.size()<<'\t'<<rsum_filter/fb_r.size()<<'\t'<<rp_n_filter/fb_r.size()<<endl;
		fout<<"mid:\t"<<mid_sum/fb_l.size()<<'\t'<<mid_p_n/fb_l.size()<<"\t"<<mid_sum_filter/fb_l.size()<<'\t'<<mid_p_n_filter/fb_l.size()<<endl;
		for (int rel=0; rel<relation_num; rel++)
		{
			int num = rel_num[rel];
			fout<<"rel:"<<id2relation[rel]<<' '<<num<<endl;
			fout<<"left:"<<lsum_r[rel]/num<<'\t'<<lp_n_r[rel]/num<<"\t"<<lsum_filter_r[rel]/num<<'\t'<<lp_n_filter_r[rel]/num<<endl;
			fout<<"right:"<<rsum_r[rel]/num<<'\t'<<rp_n_r[rel]/num<<'\t'<<rsum_filter_r[rel]/num<<'\t'<<rp_n_filter_r[rel]/num<<endl;
			fout<<"mid:"<<mid_sum_r[rel]/num<<'\t'<<mid_p_n_r[rel]/num<<'\t'<<mid_sum_filter_r[rel]/num<<'\t'<<mid_p_n_filter_r[rel]/num<<endl;
		}
		fout.close();
    }

};
Test test;

void prepare()
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	FILE* f3 = fopen("../data/word2id.txt","r");
	FILE* f4 = fopen("../data/entityWords.txt","r");
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		mid2type[st]="None";
		entity_num++;
	}
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
	//建立word2ID、ID2word map
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		word2id[st]=x;		//<word,ID>
		id2word[x]=st;		//<ID,word>
		word_num++;
	}
	//建立entityWords_vec 
	entityWords_vec.resize(entity_num);		//entityWords_vec分配空间
	while (fscanf(f4,"%s%d",buf,&x)==2)
	{
		string st=buf;		//entity
		int temp_entity_id = entity2id[st];		//entity_id
		int temp_word_num = x;
		entityWords_vec[temp_entity_id].resize(temp_word_num);
		for(int i = 0;i<temp_word_num;i++)		//x个特征词
		{
			fscanf(f4,"%s",buf);		//读入word
			string st1=buf;		//word
			entityWords_vec[temp_entity_id][i] = word2id[st1];		//build entityWords_vec
		}
	}
    FILE* f_kb = fopen("../data/test.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
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
        	cout<<"miss relation:"<<s3<<endl;
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen("../data/train.txt","r");
	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
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

        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen("../data/valid.txt","r");
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
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
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
        if (argc==3)
            trainortest = argv[2];
        prepare();
        test.run();
    }
}

