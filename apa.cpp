#include<bits/stdc++.h>
using namespace std;
void query1(long long m){
	cout <<"? 1" <<" " << m << endl;
	return;
}
void query2(long long l,long long r){
	cout <<"? 2" << " " << l << " " << r <<endl;
	return;
}
void solve(){
	long long n,i,j,prev,ne,lower,l,r,mid;
	long long x[1001][1001];
	vector<long long> v,q[1001];
	set<char> s[1001][1001];
	vector<char> ans;
	char p;
	cin >> n;
	ans.resize(n);
	prev=0;
	for(i=1;i<=n;i++){
		query2(1,i);
		cin >> ne;
		if(ne>prev){
			prev=ne;
			v.push_back(i);
		}
	}
	for(j=0;j<v.size();j++){
		i=v[j];
		query1(i);
		cin >> p;
		ans[i-1]=p;
	}
	for(i=1;i<=n;i++){
		lower=lower_bound(v.begin(),v.end(),i)-v.begin();
		if(lower!=(long long)v.size()&&v[lower]==i){
			x[i][i]=1;
			s[i][i].insert(ans[i-1]);
			q[i].push_back(i);
			for(j=i-1;j>0;j--){
				if(s[i][j+1].count(ans[j-1])){
					s[i][j]=s[i][j+1];
					x[i][j]=x[i][j+1];
				}
				else{
					s[i][j]=s[i][j+1];
					x[i][j]=x[i][j+1]+1;
					s[i][j].insert(ans[j-1]);
					q[i].push_back(j);
				}
			}
			continue;
		}
		l=0;
		r=(long long)q[i-1].size()-1;
		while(l<r){
			mid=(l+r)/2;
			query2(q[i-1][mid],i);
			cin >> ne;
			if(ne==mid+1){
				r=mid;
			}
			else{
				l=mid+1;
			}
		}
		ans[i-1]=ans[q[i-1][l]-1];
		x[i][i]=1;
		s[i][i].insert(ans[i-1]);
		q[i].push_back(i);
		for(j=i-1;j>0;j--){
			if(s[i][j+1].count(ans[j-1])){
				s[i][j]=s[i][j+1];
				x[i][j]=x[i][j+1];
			}
			else{
				s[i][j]=s[i][j+1];
				x[i][j]=x[i][j+1]+1;
				s[i][j].insert(ans[j-1]);
				q[i].push_back(j);
			}
		}
	}
	cout << "! ";
	for(i=0;i<n;i++){
		cout << ans[i];
	}
}
signed main(){
    long long t;
  	t=1;
    while(t--){
        solve();
    }
}