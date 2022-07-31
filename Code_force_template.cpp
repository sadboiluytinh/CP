//Initialize array
//Check edge cases
//LifeIsGood


#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<ll> vll;
typedef pair<ll, ll> pll;
#define MOD 1000000007
#define nor 100007
#define N 200001

queue<int> q;
ll dist[N];
vector<ll> adj[N];
bool visited[N];

ll gcd(ll a, ll b)
{
    if (b == 0)
    return a;
    return gcd(b, a % b);
}
    
ll lcm(ll a, ll b)
{
    return (a / gcd(a, b)) * b;
}

ll dfs(ll s, ll p)
{
    ll sum = 0;
    ll z = 0;
    
    for(auto x: adj[s])
    {
        if(x!=p)
        {
            ll t = dfs(x,s);
            sum+=t;
            if(t==0)
            {
                z++;
            }
        }
    }
    
    return sum+max(0ll,z-1);
}

void bfs(int x)
{
    visited[x] = true;
    dist[x] = 0;
    q.push(x);
    while (!q.empty())
    {
        int s = q.front();
        q.pop();
        // process node s (Do something here)
        for (auto u : adj[s])
        {
            if (visited[u]) continue;
            visited[u] = true;
            dist[u] = dist[s]+1;
            q.push(u);
        }
    }
}
vector<pll> AdjList[N]; // Weight and outnode
vll dfs_num; // Init: 0 for unvisited, 1 otherwise
vll ts; // global vector to store the toposort in reverse order
void dfs2(ll u)
{ // different function name compared to the original dfs
    dfs_num[u] = 1;
    for (int j = 0; j < (int)AdjList[u].size(); j++)
    {
        pll v = AdjList[u][j];
        if (dfs_num[v.first] == 0) dfs2(v.first);
        if(dfs_num[v.first]==1)
        {
            cout << "Cycle" << endl;
            return;
        }
    }
    ts.push_back(u);
}

struct DSU
{
    vll parents, sizes, sum;
    DSU(ll n) : parents(n), sizes(n), sum(n) {};
    
    void Make(ll v)
    {
        parents[v] = v;
        sizes[v] = 1;
    }
    
    ll Root(ll v)
    {
        return parents[v] == v ? v : parents[v] = Root(parents[v]);
    }
    
    bool SameSet(ll v_0, ll v_1)
    {
        return Root(v_0) == Root(v_1);
    }
    
    bool Merge(int v_0, int v_1)
    {
        if ((v_0 = Root(v_0)) == (v_1 = Root(v_1)))
        {
            return false;
        }
        if (sizes[v_0] > sizes[v_1])
        {
            swap(v_0, v_1);
        }
        parents[v_0] = v_1;
        sizes[v_1] += sizes[v_0];
        return true;
    }
};

struct FT
{
    private: vll ft; // recall that vll is: typedef vector<long long> vll;
    public: FT(int n) { ft.assign(n + 1, 0); } // init n + 1 zeroes
    int rsq(int b)
    { // returns RSQ(1, b)
        int sum = 0;
        for (; b; b -= (b&-b)) sum += ft[b];
        return sum;
    }
    int rsq(int a, int b)
    { // returns RSQ(a, b)
        return rsq(b) - (a == 1 ? 0 : rsq(a - 1));
    }
    // adjusts value of the k-th element by v (v can be +ve/inc or -ve/dec)
    void adjust(int k, int v)
    { // note: n = ft.size() - 1
        for (; k < (int)ft.size(); k += (k&-k)) ft[k] += v;
    }
};


struct segment_tree
{
    ll k; 
    vll seg;
    vll arr;
    segment_tree(ll n)
    {
        k = n+10;
        seg.resize(4*k+1);
        arr.resize(k+1);
    }

    ll merge(ll a , ll b )
    {
        return (a+b);   // change here according to question
    }

    void build(ll ind , ll low , ll high)
    {
        if(low == high){
                seg[ind] = arr[low];
                return;
        }
        ll mid = (low + high)>>1;
        build(2*ind+1 , low , mid);
        build(2*ind+2 , mid+1 , high);
        seg[ind] = merge(seg[2*ind+1] , seg[2*ind+2]);
    }

    ll query(ll ind , ll low , ll high , ll l, ll r)
    {
        if(low >= l && high <= r) return seg[ind];
        if(high < l || low > r) return 0;    // change here according to question
        ll mid = (low + high)>>1;
        ll left = query(2*ind+1 , low , mid , l , r);
        ll right = query(2*ind+2 , mid+1 , high , l , r);
        return merge(left , right);
    }

    void update(ll ind , ll low , ll high , ll node , ll val)
    {
        if(low == high)
        {
                seg[ind] = val;
                arr[node] = val;
                return;
        }
        else
        {
                ll mid = (low + high)>>1;
                if(node<=mid)
                {
                update(2*ind+1 , low , mid , node , val);
                }
                else update(2*ind+2 , mid+1 , high , node , val);
                seg[ind] = merge(seg[2*ind+1] , seg[2*ind+2]);
        }
    }

    ll query(ll l , ll r)
    {
        return query(0,0,N-1,l,r);
    }
    void update(ll pos, ll val)
    {
        update(0,0,N-1,pos,val);
    }
};


vll dfs_low, S; // global variables
ll dfsNumberCounter = 0;
ll numSCC = 0;
void tarjanSCC(int u)
{
    dfs_low[u] = dfs_num[u] = dfsNumberCounter++; // dfs_low[u] <= dfs_num[u]
    S.push_back(u); // stores u in a vector based on order of visitation
    visited[u] = 1;
    for (int j = 0; j < (int)AdjList[u].size(); j++)
    {
        pll v = AdjList[u][j];
        if (dfs_num[v.first] == 0) tarjanSCC(v.first);
        if (visited[v.first]) // condition for update
        {
            dfs_low[u] = min(dfs_low[u], dfs_low[v.first]);
        }
    }
    if (dfs_low[u] == dfs_num[u]) { // if this is a root (start) of an SCC
    printf("SCC %d:", ++numSCC); // this part is done after recursion
    while (1)
    {
        int v = S.back(); S.pop_back(); visited[v] = 0;
        printf(" %d", v);
        if (u == v) break;
    }
    printf("\n");
    }
}

vll arr;
void dfs_traverse(ll s)
{
    if (visited[s]) return;
    arr.push_back(s);
    visited[s] = true;
    for (ll u: adj[s])
    {
        dfs_traverse(u);
    }
}

void Kadane(ll a[], ll size)
{
    ll ans = INT_MIN, pre = 0, l =0, r = 0, s=0;
 
    for (ll i=0; i< size; i++ )
    {
        pre += a[i];
 
        if (ans < pre)
        {
            ans = pre;
            l = s;
            r = i;
        }
 
        if (pre < 0)
        {
            pre = 0;
            s = i + 1;
        }
    }
    cout << ans << endl;
    cout << l << " " << r << endl;
}
pll getpow(ll n, ll x)
{
    ll ans = 0;
    while(n%x==0)
    {
        ans++;
        n/=x;
    }
    
    return {ans,n};
}
void solve()
{
    ll n;
    cin >> n;
    cout << 2 << endl;
    for(ll i=1; i<=n; i++)
    {
        if(i%2==1)
        {
            ll x = i;
            while(x<=n)
            {
                cout << x << " ";
                x*=2;
            }
        }
    }

    cout << endl;
}


signed main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int tc;
    cin >> tc;
    for(int i=1; i<=tc; i++)
    {
        solve();
    }
    // solve();
}