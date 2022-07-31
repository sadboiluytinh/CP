#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<ll> vll;
typedef pair<ll, ll> pll;
#define MOD 1000000007
class SegmentTree
{ // the segment tree is stored like a heap array
    private: vll st, A; // recall that vll is: typedef vector<long long> vll;
    int n;
    int left (int p) { return p << 1; } // same as binary heap operations
    int right(int p) { return (p << 1) + 1; }
    void build(int p, int L, int R)
    { // O(n)
        if (L == R) // as L == R, either one is fine
            st[p] = L; // store the index
        else
        { // recursively compute the values
            build(left(p) , L , (L + R) / 2);
            build(right(p), (L + R) / 2 + 1, R );
            int p1 = st[left(p)], p2 = st[right(p)];
            st[p] = (A[p1] <= A[p2]) ? p1 : p2;
        }
    }
    int rmq(int p, int L, int R, int i, int j)
    { // O(log n)
        if (i > R || j < L) return -1; // current segment outside query range
        if (L >= i && R <= j) return st[p]; // inside query range
        // compute the min position in the left and right part of the interval
        int p1 = rmq(left(p) , L , (L+R) / 2, i, j);
        int p2 = rmq(right(p), (L+R) / 2 + 1, R , i, j);
        if (p1 == -1) return p2; // if we try to access segment outside query
        if (p2 == -1) return p1; // same as above
        return (A[p1] <= A[p2]) ? p1 : p2; // as in build routine
    }

    public:
    SegmentTree(const vll &_A)
    {
        A = _A; n = (int)A.size(); // copy content for local usage
        st.assign(4 * n, 0); // create large enough vector of zeroes
        build(1, 0, n - 1); // recursive build
    }

    int rmq(int i, int j) { return rmq(1, 0, n - 1, i, j); } // overloading
};

class FenwickTree
{
    private: vll ft; // recall that vll is: typedef vector<long long> vll;
    public: FenwickTree(int n) { ft.assign(n + 1, 0); } // init n + 1 zeroes
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

//Declare N as a constant
vector<ll> adj[N];
bool visited[N];

void dfs(int s)
{
    if (visited[s]) return;
    visited[s] = true;
    // process node s (Do something here)
    for (auto u: adj[s])
    {
        dfs(u);
    }
}

//Declare N as a constant
queue<int> q;
bool visited[N];
ll dist[N];
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

ll numofconnected()
{
    ll ans = 0;
    for(int i=0; i<N; i++) // change for any graphs
    {
        if(!visited[i])
        {
            ans++;
            bfs(i);
        }
    }
}


ll color[MOD];
bool checkBipartiteGraph() {
    fill_n(color, n + 1, -1);
    queue <int> q;
    q.push(0); // Start with some degree
    color[0] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (auto v : adj[u]) {
            if (color[v] == color[u]) return false;
            if (color[v] == -1)
            {
                color[v] = !color[u]; // Change color on adjacent list
                q.push(v);
            }
        }
    }
    return true;
}

int num[]; //cho biết thứ tự duyệt DFS của các đỉnh (thứ tự mà mỗi đỉnh bắt đầu duyệt).
int low[]; // low[u] cho biết thứ tự 
//(giá trị num) nhỏ nhất có thể đi đến được từ u bằng cách đi xuôi xuống theo các cạnh nét liền
//(các cung trên cây DFS) và kết thúc đi ngược lên không quá 1 lần theo cạnh nét đứt. 

int tail[]; //cho biết thời điểm kết thúc duyệt DFS của mỗi đỉnh cũng là thời điểm duyệt xong của đỉnh đó .
int timeDfs = 0; // Thứ tự duyệt DFS
//DFS Tree
void dfs(int u, int pre) {
    num[u] = low[u] = ++timeDfs;
    for (int v : g[u]){
        if (v == pre) continue;
        if (!num[v]) {
            dfs(v, u);
            low[u] = min(low[u], low[v]);
        }
        else low[u] = min(low[u], num[v]);
    }
    tail[u] = timeDfs;
}

int V = 10000000;
vector<pair<int,int>> AdjList[]; // Pair, with the first is the destination and the second is the length

void BellmanFord(int s)
{
    vector<int> dist(V, 1e9+7);
    dist[s] = 0;
    for (int i = 0; i < V - 1; i++) // relax all E edges V-1 times
    {
        for (int u = 0; u < V; u++) // these two loops = O(E), overall O(VE)
        {
            for (int j = 0; j < (int)AdjList[u].size(); j++)
            {
                pair<int,int> v = AdjList[u][j]; // record SP spanning here if needed
                dist[v.first] = min(dist[v.first], dist[u] + v.second); // relax
            }
        }
    }
 
    bool hasNegativeCycle = false;
    for (int u = 0; u < V; u++) // one more pass to check
    {
        for (int j = 0; j < (int)AdjList[u].size(); j++)
        {
            pair<int,int> v = AdjList[u][j];
            if (dist[v.first] > dist[u] + v.second) // if this is still possible
            hasNegativeCycle = true; // then negative cycle exists!
        }
    }
    return;
}

int n;
void SPFA(int S)
{
    vector<int> dist(n, MOD); dist[S] = 0;
    queue<int> q; q.push(S);
    vector<int> in_queue(n, 0); in_queue[S] = 1;
    while (!q.empty())
    {
        int u = q.front(); q.pop(); in_queue[u] = 0;
        for (int j = 0; j < (int)AdjList[u].size(); j++)
        { // all neighbors of u
            int v = AdjList[u][j].first, weight_u_v = AdjList[u][j].second;
            if (dist[u] + weight_u_v < dist[v])
            { // if can relax
                dist[v] = dist[u] + weight_u_v; // relax
                if (!in_queue[v])
                { // add to the queue
                    q.push(v); // only if it is not already in the queue
                    in_queue[v] = 1;
                }
            }
        }
    }
}

const long long INF = 2000000000000000000LL;
struct Edge
{
    int u, v;
    long long w; // cạnh từ u đến v, trọng số w
};
void bellmanFord(int n, int S, vector<Edge> &e, vector<long long> &D, vector<int> &trace)
{
    // e: danh sách cạnh
    // n: số đỉnh
    // S: đỉnh bắt đầu
    // D: độ dài đường đi ngắn nhất
    // trace: mảng truy vết đường đi
    // INF nếu không có đường đi
    // -INF nếu có đường đi âm vô tận
    D.resize(n, INF);
    trace.resize(n, -1);

    D[S] = 0;
    for(int T = 1; T < n; T++)
    {
        for (auto E : e)
        {
            int u = E.u;
            int v = E.v;
            long long w = E.w;
            if (D[u] != INF && D[v] > D[u] + w)
            {
                D[v] = D[u] + w;
                trace[v] = u;
            }
        }
    }
}

vector<int> trace_path(vector<int> &trace, int S, int u)
{
    if (u != S && trace[u] == -1) return vector<int>(0); // không có đường đi

    vector<int> path;
    while (u != -1)
    { // truy vết ngược từ u về S
        path.push_back(u);
        u = trace[u];
    }
    reverse(path.begin(), path.end()); // cần reverse vì đường đi lúc này là từ u về S
    
    return path;
}

bool findNegativeCycle(int n, vector<long long> &D, vector<int> &trace, vector<int> &negCycle,vector<Edge> &e) {
    // mảng D và trace đã được chạy qua thuật toán Bellman-Ford
    int negStart = -1; // đỉnh bắt đầu
    for (auto E : e)
    {
        int u = E.u;
        int v = E.v;
        long long w = E.w;
        if (D[u] != INF && D[v] > D[u] + w)
        {
            D[v] = -INF; 
            trace[v] = u;
            negStart = v; // đã tìm thấy -INF
        }
    }

    if (negStart == -1) return false; // không có chu trình âm

    int u = negStart;
    for (int i = 0; i < n; i++)
    {
        u = trace[u]; // đưa u về chu trình âm
    }

    negCycle = vector<int>(1, u);
    for (int v = trace[u]; v != u; v = trace[u])
    {
        negCycle.push_back(v); // truy vết một vòng
    }
    reverse(negCycle.begin(), negCycle.end());
    return true;
}



const long long INF = 2000000000000000000LL;
struct Edge
{
    int v;
    long long w;
};
void dijkstra(int n, int S, vector<vector<Edge>> E, vector<long long> &D, vector<int> &trace) 
{
    D.resize(n, INF);
    trace.resize(n, -1);
    
    vector<bool> P(n, 0);
    D[S] = 0;
    
    for (int i = 0; i < n; i++)
    {
        int uBest; // tìm đỉnh u chưa dùng, có khoảng cách nhỏ nhất
        long long Max = INF;
        for (int u = 0; u < n; u++)
        {
            if(D[u] < Max && P[u] == false)
            {
                uBest = u;
                Max = D[u];
            }
        }
        
        // cải tiến các đường đi qua u
        int u = uBest;
        P[u] = true;
        for(auto x : E[u])
        {
            int v = x.v;
            long long w = x.w;
            if(D[v] > D[u] + w)
            {
                D[v] = D[u] + w;
                trace[v] = u;
            }
        }
    }
}

void Dijkstra_with_pq(int s)
{
    vector<int> dist(V, INF);
    dist[s] = 0; // INF = 1B to avoid overflow
    priority_queue< pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>> > pq;
    pq.push(pair<int,int>(0, s));
    while (!pq.empty())
    { // main loop
        pair<int,int> front = pq.top();
        pq.pop(); // greedy: get shortest unvisited vertex
        int d = front.first, u = front.second;
        if (d > dist[u]) continue; // this is a very important check
        for (int j = 0; j < (int)AdjList[u].size(); j++)
        {
        pair<int,int> v = AdjList[u][j]; // all outgoing edges from u
        if (dist[u] + v.second < dist[v.first])
            {
            dist[v.first] = dist[u] + v.second; // relax operation
            pq.push(pair<int,int>(dist[v.first], v.first));
            }
        }
    } // this variant can cause duplicate items in the priority queue
}

vector<int> trace_path(vector<int> &trace, int S, int u)
{
    if (u != S && trace[u] == -1) return vector<int>(0); // không có đường đi

    vector<int> path;
    while (u != -1)
    { // truy vết ngược từ u về S
        path.push_back(u);
        u = trace[u];
    }
    reverse(path.begin(), path.end()); // cần reverse vì đường đi lúc này là từ u về S
    
    return path;
}

void init_trace(vector<vector<int>> &trace)
{
    int n = trace.size();
    for (int u = 0; u < n; u++)
    {
        for (int v = 0; v < n; v++)
        {
            trace[u][v] = u;
        }
    }
}

void floydWarshall(int n, vector<vector<long long>> &w, vector<vector<long long>> &D, vector<vector<int>> &trace)
{
    D = w;
    init_trace(trace); // nếu cần dò đường đi
    
    for (int k = 0; k < n; k++)
    {
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++)
            {
                //AdjMat[i][j] |= (AdjMat[i][k] & AdjMat[k][j]);
                //For checking connect (directly/indirectly)
                if (D[u][v] > D[u][k] + D[k][v])
                {
                    D[u][v] = D[u][k] + D[k][v];
                    trace[u][v] = trace[k][v];
                }
            }
        }
    }
}

vector<int> trace_path(vector<vector<int>> &trace, int u, int v)
{
    vector<int> path;
    while (v != u)
    { // truy vết ngược từ v về u
        path.push_back(v);
        v = trace[u][v];
    }
    path.push_back(u);
    
    reverse(path.begin(), path.end()); // cần reverse vì đường đi từ v ngược về u
    return path;
}

vector<pair<ll,pair<ll,ll>>> EdgeList;
int E;
void kruskal()
{
    // for (int i = 0; i < E; i++)
    // {
    //     EdgeList.push_back(make_pair(w, pair<int,int>(u, v)));
    // } // (w, u, v)
    sort(EdgeList.begin(), EdgeList.end()); // sort by edge weight O(E log E)
    // note: pair object has built-in comparison function
    int mst_cost = 0;
    DSU UF(V); // all V are disjoint sets initially
    for (int i = 0; i < E; i++)
    { // for each edge, O(E)
        pair<ll, pair<ll,ll>> front = EdgeList[i];
        if (!UF.SameSet(front.second.first, front.second.second))
        { // check
            mst_cost += front.first; // add the weight of e to MST
            UF.SameSet(front.second.first, front.second.second); // link them
        }
    } // note: the runtime cost of UFDS is very light
// note: the number of disjoint sets must eventually be 1 for a valid MST
    printf("MST cost = %d (Kruskal’s)\n", mst_cost);
}



vll taken; // global boolean flag to avoid cycle
priority_queue<pair<ll,ll>> pq; // priority queue to help choose shorter edges
// note: default setting for C++ STL priority_queue is a max heap
void process(ll vtx)
{ // so, we use -ve sign to reverse the sort order
    taken[vtx] = 1;
    for (int j = 0; j < (int)AdjList[vtx].size(); j++)
    {
        pair<ll,ll> v = AdjList[vtx][j];
        if (!taken[v.first]) pq.push(pll(-v.second, -v.first));
    }
    
} // sort by (inc) weight then by (inc) id
    // inside int main()---assume the graph is stored in AdjList, pq is empty
int main()
{
    taken.assign(V, 0); // no vertex is taken at the beginning
    process(0); // take vertex 0 and process all edges incident to vertex 0
    ll mst_cost = 0;
    while (!pq.empty())
    { // repeat until V vertices (E=V-1 edges) are taken
        pll front = pq.top(); pq.pop();
        ll u = -front.second;
        ll w = -front.first; // negate the id and weight again
        if (!taken[u]) // we have not connected this vertex yet
        mst_cost += w, process(u); // take u, process all edges incident to u
    } // each edge is in pq only once!
    printf("MST cost = %d (Prim’s)\n", mst_cost);
}

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

ll n;
vector<vector<ll>> adj, adj_t;
vector<ll> used;
vector<ll> order, comp;
vector<bool> assignment;

void dfs1(ll v)
{
    used[v] = true;
    for (int u : adj[v])
    {
        if (!used[u])
            dfs1(u);
    }
    order.push_back(v);
}

void dfs2(int v, int cl)
{
    comp[v] = cl;
    for (int u : adj_t[v])
    {
        if (comp[u] == -1)
            dfs2(u, cl);
    }
}

bool solve_2SAT()
{
    order.clear();
    used.assign(n, false);
    for (int i = 0; i < n; ++i)
    {
        if (!used[i])
            dfs1(i);
    }

    comp.assign(n, -1);
    for (int i = 0, j = 0; i < n; ++i)
    {
        int v = order[n - i - 1];
        if (comp[v] == -1)
            dfs2(v, j++);
    }

    assignment.assign(n / 2, false);
    for (int i = 0; i < n; i += 2)
    {
        if (comp[i] == comp[i + 1])
            return false;
        assignment[i / 2] = comp[i] > comp[i + 1];
    }
    return true;
}

void add_disjunction(int a, bool na, int b, bool nb)
{
    // na and nb signify whether a and b are to be negated 
    a = 2*a ^ na;
    b = 2*b ^ nb;
    int neg_a = a ^ 1;
    int neg_b = b ^ 1;
    adj[neg_a].push_back(b);
    adj[neg_b].push_back(a);
    adj_t[b].push_back(neg_a);
    adj_t[a].push_back(neg_b);
}
//Use AdjMat: adj[u][v] = x if u->v is connected with capacity x
bool bfs_ford(ll adjMat[100005][100005], ll s, ll t, ll parent[])
{
    bool visited[V];
    memset(visited, 0, sizeof(visited));

    queue<ll> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while (!q.empty())
    {
        ll u = q.front();
        q.pop();

        for (ll v = 0; v < 100005; v++)
        {
            if (visited[v] == false && adjMat[u][v] > 0)
            {
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }

    return (visited[t] == true);
}

ll fordFulkerson(int graph[100005][10005], int s, int t)
{
  ll u, v;

  ll rGraph[100005][100005];
  for (u = 0; u < 100005; u++)
    for (v = 0; v < 100005; v++)
      rGraph[u][v] = graph[u][v];
  ll parent[V];
  ll max_flow = 0;

  // Updating the residual values of edges
  while (bfs_ford(rGraph, s, t, parent))
  {
    ll path_flow = INT_MAX;
    for (v = t; v != s; v = parent[v])
    {
        u = parent[v];
        path_flow = min(path_flow, rGraph[u][v]);
    }

    for (v = t; v != s; v = parent[v])
    {
        u = parent[v];
        rGraph[u][v] -= path_flow;
        rGraph[v][u] += path_flow;
    }

    // Adding the path flows
    max_flow += path_flow;
  }
  return max_flow;
}

bool prime(ll n)
{
    if (n < 2) return false;
    for (ll x = 2; x*x <= n; x++)
    {
        if (n%x == 0) return false;
    }
    return true;
}

vll factors(ll n)
{
    vll f;
    for (ll x = 2; x*x <= n; x++)
    {
        while (n%x == 0)
        {
        f.push_back(x);
        n /= x;
        }
    }
    if (n > 1) f.push_back(n);
    return f;
}

int modpow(ll x, ll n, ll m)
{
    if (n == 0) return 1%m;
    long long u = modpow(x,n/2,m);
    u = (u*u)%m;
    if (n%2 == 1) u = (u*x)%m;
    return u;
}



#define N 1000000
void cof(int mat[N][N], int temp[N][N], int p, int q, int n)
{
    int i = 0, j = 0;
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (row != p && col != q)
            {
                temp[i][j++] = mat[row][col];
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

int det(int mat[N][N], int n)
{
    int D = 0;
    if (n == 1)
        return mat[0][0];
 
    int temp[N][N]; 
 
    int sign = 1;
    for (int f = 0; f < n; f++)
    {
        cof(mat, temp, 0, f, n);
        D += sign * mat[0][f] * det(temp, n - 1);
        sign = -sign;
    }
 
    return D;
}