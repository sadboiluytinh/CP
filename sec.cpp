#include<bits/stdc++.h>
using namespace std;
 
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template <typename num_t>
using ordered_set = tree<num_t, null_type, less_equal<num_t>, rb_tree_tag, tree_order_statistics_node_update>;
//priority_queue<pair<int,int>, vector<pair<int,int> > , greater<pair<int,int> > > p;
 
#define pb                     push_back
#define pie                    3.1415926535
#define inf                    9e18
#define mod                    1000000007
#define all(x)                 (x).begin(),(x).end()
#define int                    long long
#define endl                   '\n'
#define debug(x)               cout << '>' << #x << ':' << x << endl;
#define printKick(a,b)         cout << "Case #"<<a<<": "<<b<<endl;
 
void fastio(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
}
/****************************************/
vector<int> solve(int k){
    string s = to_string(k);
    set<char> S;
    for(auto value : s){
        S.insert(value);
    }
    vector<int> v;
    for(auto value : S){
        if(value == '0' || value == '1')continue;
        else{
            v.pb((value - '0'));
        }
    }
 
    int f = v.size();
    for(int i = 0;i<f;i++){
        v[i] *= k;
    }
 
    return v;
}
int32_t main() {
    fastio();
    int n,x;
    cin>>n>>x;
    string s = to_string(x);
    int p = s.size();
    if(p == n){
        cout<<0<<endl;
        return 0;
    }
    queue<pair<int,int> > q;
    q.push({x,0});
    int ans = -1;
    set<int> taken;
    taken.insert(x);
    while(!q.empty()){
        pair<int,int> curr = q.front();
        q.pop();
        vector<int> x = solve(curr.first);
        int l = curr.second;
        bool can = false;
        for(auto value : x){
            if(taken.find(value) != taken.end())continue;
            string g = to_string(value);
            if(g.size() == n){
                can = true;
                ans = l + 1;
                break;
            }
            q.push({value, l + 1});
            taken.insert(value);
        }
        if(can)break;
    }
    cout<<ans<<endl;
    
}