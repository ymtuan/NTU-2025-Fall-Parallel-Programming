// Compile: g++ -std=c++17 -O3 -ltbb hw1_tbb.cpp -o hw1
// Execute: srun -A ACD114118 -n1 -c${threads} ./hw1 ${input}

#include <bits/stdc++.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
using namespace std;

const int MAX_CELLS = 256;
const int INF = 1000000007;

struct State {
    int player;
    bitset<MAX_CELLS> boxes;
};
struct StateHash {
    size_t operator()(const State& s) const {
        size_t h = std::hash<int>{}(s.player);
        const unsigned long long* data = (const unsigned long long*)&s.boxes;
        constexpr int N_WORDS = MAX_CELLS / (sizeof(unsigned long long) * 8);
        for (int i = 0; i < N_WORDS; ++i)
            h ^= std::hash<unsigned long long>{}(data[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
struct StateEqual {
    bool operator()(const State& a, const State& b) const {
        return a.player == b.player && a.boxes == b.boxes;
    }
};

struct BoxesHash {
    size_t operator()(const bitset<MAX_CELLS>& bs) const {
        size_t h = 0;
        const unsigned long long* data = (const unsigned long long*)&bs;
        constexpr int N_WORDS = MAX_CELLS / (sizeof(unsigned long long) * 8);
        for (int i = 0; i < N_WORDS; ++i)
            h ^= std::hash<unsigned long long>{}(data[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
struct BoxesEqual {
    bool operator()(const bitset<MAX_CELLS>& a, const bitset<MAX_CELLS>& b) const {
        return a == b;
    }
};

int rows=0, cols=0;
vector<bool> walls(MAX_CELLS,false), targets(MAX_CELLS,false), fragile(MAX_CELLS,false), dead_zone(MAX_CELLS,false);
bitset<MAX_CELLS> target_bits;
vector<pair<int,int>> id_to_rc(MAX_CELLS);
int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
char dir_char[4] = {'W','S','A','D'};

inline bool in_bounds_idx(int idx) { return idx >=0 && idx < rows*cols; }

// Deadlock checks
bool is_deadlock_simple(const State &s) {
    for (size_t i = s.boxes._Find_first(); i < MAX_CELLS; i = s.boxes._Find_next(i)) {
        if (targets[i]) continue;
        auto [r,c] = id_to_rc[i];
        auto blocked = [&](int rr,int cc)->bool{
            if(rr<0||rr>=rows||cc<0||cc>=cols) return true;
            int idx = rr*cols+cc;
            return walls[idx];
        };
        bool up=blocked(r-1,c), down=blocked(r+1,c), left=blocked(r,c-1), right=blocked(r,c+1);
        if ((up||down)&&(left||right)) return true;
    }
    return false;
}
bool is_deadlock(const State &s) {
    if(is_deadlock_simple(s)) return true;
    for(size_t i = s.boxes._Find_first(); i < MAX_CELLS; i = s.boxes._Find_next(i)) {
        if(targets[i]) continue;
        if(dead_zone[i]) return true;
        auto [r,c] = id_to_rc[i];
        auto is_wall_blocked = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            return walls[rr * cols + cc];
        };
        auto is_blocked_any = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            int idx = rr * cols + cc;
            return walls[idx] || s.boxes[idx];
        };

        if (is_wall_blocked(r, c-1) && is_wall_blocked(r, c+1)) {
            if (is_blocked_any(r-1, c) && is_blocked_any(r+1, c)) return true;
        }
        if (is_wall_blocked(r-1, c) && is_wall_blocked(r+1, c)) {
            if (is_blocked_any(r, c-1) && is_blocked_any(r, c+1)) return true;
        }
    }
    return false;
}

// Reachability & BFS path
bitset<MAX_CELLS> player_reachable(int start, const bitset<MAX_CELLS> &boxes) {
    bitset<MAX_CELLS> seen; seen.reset();
    queue<int> q;
    seen[start]=1; q.push(start);
    while(!q.empty()){
        int u=q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for(int d=0;d<4;++d){
            int nr=r+dr[d], nc=c+dc[d];
            if(nr<0||nr>=rows||nc<0||nc>=cols) continue;
            int v = nr*cols+nc;
            if(walls[v] || boxes[v]) continue;
            if(!seen[v]) { seen[v]=1; q.push(v); }
        }
    }
    return seen;
}
string bfs_player_path(const State &s,int from_idx,int to_idx){
    if(from_idx==to_idx) return string();
    int N = rows*cols;
    vector<int> prev(N,-1), prev_dir(N,-1);
    queue<int> q; bitset<MAX_CELLS> seen; seen.reset();
    q.push(from_idx); seen[from_idx]=1;
    bool found=false;
    while(!q.empty() && !found){
        int u=q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for(int d=0;d<4;++d){
            int nr=r+dr[d], nc=c+dc[d];
            if(nr<0||nr>=rows||nc<0||nc>=cols) continue;
            int v=nr*cols+nc;
            if(walls[v] || s.boxes[v]) continue;
            if(seen[v]) continue;
            seen[v]=1; prev[v]=u; prev_dir[v]=d;
            if(v==to_idx){ found=true; break; }
            q.push(v);
        }
    }
    if(!found) return string();
    vector<char> steps;
    int cur=to_idx;
    while(cur!=from_idx){
        steps.push_back(dir_char[prev_dir[cur]]);
        cur=prev[cur];
    }
    reverse(steps.begin(), steps.end());
    return string(steps.begin(), steps.end());
}

// Heuristic
vector<int> manhattan_target_r, manhattan_target_c;
int heuristic_sum(const State &s){
    int N=rows*cols;
    vector<pair<int,int>> box_positions;
    box_positions.reserve(64);
    for(size_t i = s.boxes._Find_first(); i < MAX_CELLS; i = s.boxes._Find_next(i)) {
        box_positions.push_back(id_to_rc[i]);
    }
    int n = box_positions.size();
    if (n == 0) return 0;
    const int MAX_N = 64;
    int A[MAX_N + 1][MAX_N + 1] = {0};
    for(int i=0; i<n; ++i){
        auto [br, bc] = box_positions[i];
        for(int j=0; j<n; ++j){
            int d = abs(br - manhattan_target_r[j]) + abs(bc - manhattan_target_c[j]);
            A[i+1][j+1] = d;
        }
    }
    int u[MAX_N + 1] = {0}, v[MAX_N + 1] = {0}, p[MAX_N + 1] = {0}, way[MAX_N + 1] = {0};
    for (int i=1; i<=n; ++i) {
        p[0] = i;
        int j0 = 0;
        int minv[MAX_N + 1];
        bool used[MAX_N + 1] = {false};
        for (int k = 0; k <= n; ++k) minv[k] = INF;
        do {
            used[j0] = true;
            int i0 = p[j0], delta = INF, j1;
            for (int j=1; j<=n; ++j)
                if (!used[j]) {
                    int cur = A[i0][j] - u[i0] - v[j];
                    if (cur < minv[j])
                        minv[j] = cur, way[j] = j0;
                    if (minv[j] < delta)
                        delta = minv[j], j1 = j;
                }
            for (int j=0; j<=n; ++j)
                if (used[j])
                    u[p[j]] += delta, v[j] -= delta;
                else
                    minv[j] -= delta;
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    return -v[0];
}

// Reconstruct moves
string reconstruct_full_moves(const unordered_map<State, pair<State, char>, StateHash, StateEqual> &parent, const State &goal){
    vector<State> states;
    State cur = goal;
    while(true){
        states.push_back(cur);
        auto it = parent.find(cur);
        if(it == parent.end()) break;
        State pk = it->second.first;
        if(pk.player == -1) break;
        cur = pk;
    }
    reverse(states.begin(), states.end());
    string result;
    for(size_t i=0;i+1<states.size();++i){
        State ps = states[i];
        State cs = states[i+1];
        bitset<MAX_CELLS> diff = ps.boxes ^ cs.boxes;
        size_t b = diff._Find_first();
        size_t t = diff._Find_next(b);
        if(!ps.boxes[b]) std::swap(b, t);
        auto [br,bc]=id_to_rc[b]; auto [tr,tc]=id_to_rc[t];
        int d=-1;
        for(int dd=0;dd<4;++dd) if(br+dr[dd]==tr && bc+dc[dd]==tc){ d=dd; break; }
        if(d==-1){
            auto it = parent.find(cs);
            if(it != parent.end()) {
                char mv = it->second.second;
                if(mv) result.push_back(mv);
            }
            continue;
        }
        int pr=br-dr[d], pc=bc-dc[d]; int pidx=pr*cols+pc;
        string walk=bfs_player_path(ps, ps.player, pidx);
        result+=walk; result.push_back(dir_char[d]);
    }
    return result;
}

// Beam A* solver (TBB + per-thread bounded heaps + nth_element)
pair<int,string> astar_push_solver(const State &start, int beam_width){
    int N=rows*cols;
    manhattan_target_r.clear(); manhattan_target_c.clear();
    for(int i=0;i<N;++i) if(targets[i]){ manhattan_target_r.push_back(id_to_rc[i].first); manhattan_target_c.push_back(id_to_rc[i].second); }

    struct Node {
        int f,g; State s;
        bool operator<(const Node &o) const { if(f!=o.f) return f>o.f; return g<o.g; }
    };

    struct PQN { int f,g; State s; };
    struct PQCmp { bool operator()(const PQN &a, const PQN &b) const { if(a.f!=b.f) return a.f > b.f; return a.g < b.g; } };

    priority_queue<PQN, vector<PQN>, PQCmp> pq;
    unordered_map<State,int,StateHash,StateEqual> best_g;
    unordered_map<State,pair<State,char>,StateHash,StateEqual> parent;

    tbb::enumerable_thread_specific<unordered_map<bitset<MAX_CELLS>, int, BoxesHash, BoxesEqual>> heuristic_caches;
    int h0 = heuristic_sum(start);

    best_g.reserve(beam_width * 20);
    parent.reserve(beam_width * 20);
    best_g[start]=0;
    parent[start] = {State{-1, bitset<MAX_CELLS>{}}, 0};

    pq.push(PQN{h0,0,start});
    long long last_raw_per_node = 0;

    while(!pq.empty()){
        vector<PQN> current_level;
        while(!pq.empty()){
            current_level.push_back(pq.top());
            pq.pop();
        }
        if(current_level.empty()) break;

        for (const auto& cur : current_level) {
            State s = cur.s;
            bool done = (s.boxes & ~target_bits).none();
            if(done) {
                return {cur.g, reconstruct_full_moves(parent, s)};
            }
        }

        long long raw_generated = 0;

        // Candidate type: pair<Node, pair<parentState, actionChar>>
        using Cand = pair<Node, pair<State, char>>;

        // global concurrent collector
        tbb::concurrent_vector<Cand> candidates;

        // per-thread cap (try 2x beam_width)
        const size_t per_thread_cap = std::max((size_t)1, (size_t)beam_width * 2);

        // comparator for local max-heap (top = worst = largest f)
        auto local_cmp = [](const Cand &a, const Cand &b){ return a.first.f < b.first.f; };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, current_level.size()), [&](const tbb::blocked_range<size_t> &r){
            std::priority_queue<Cand, vector<Cand>, decltype(local_cmp)> local_pq(local_cmp);
            for(size_t ii = r.begin(); ii != r.end(); ++ii){
                PQN cur = current_level[ii];
                State s = cur.s; int g = cur.g;

                auto itg = best_g.find(s);
                if(itg==best_g.end() || itg->second != g) continue; // not current best

                bool done = (s.boxes & ~target_bits).none();
                if(done) continue;

                bitset<MAX_CELLS> reach = player_reachable(s.player, s.boxes);
                auto &local_cache = heuristic_caches.local();

                for(size_t b = s.boxes._Find_first(); b < MAX_CELLS; b = s.boxes._Find_next(b)){
                    auto [br,bc] = id_to_rc[b];
                    for(int d=0; d<4; ++d){
                        int pr = br - dr[d], pc = bc - dc[d], tr = br + dr[d], tc = bc + dc[d];
                        if(pr<0||pr>=rows||pc<0||pc>=cols) continue;
                        if(tr<0||tr>=rows||tc<0||tc>=cols) continue;
                        int pidx = pr*cols + pc, tidx = tr*cols + tc;
                        if(!reach[pidx]) continue;
                        if(walls[tidx] || s.boxes[tidx] || fragile[tidx] || dead_zone[tidx]) continue;
                        State ns = s; ns.boxes[b]=0; ns.boxes[tidx]=1; ns.player=b;
                        if(is_deadlock(ns)) continue;
                        int ng = g + 1;
                        int h;
                        auto hit = local_cache.find(ns.boxes);
                        if(hit != local_cache.end()) h = hit->second;
                        else { int ch = heuristic_sum(ns); h = ch; local_cache[ns.boxes] = h; }
                        Node nnode{ng+h, ng, ns};
                        Cand cand = { std::move(nnode), {s, dir_char[d]} };

                        if(local_pq.size() < per_thread_cap) local_pq.push(std::move(cand));
                        else {
                            const Cand &worst = local_pq.top();
                            if(cand.first.f < worst.first.f){
                                local_pq.pop();
                                local_pq.push(std::move(cand));
                            }
                        }
                    }
                }
            }
            while(!local_pq.empty()){
                candidates.push_back(local_pq.top());
                local_pq.pop();
            }
        }); // end parallel_for

        raw_generated = candidates.size();

        // Move concurrent_vector -> sequential vector
        vector<Cand> seq_candidates;
        seq_candidates.reserve(raw_generated);
        for(auto &c : candidates) seq_candidates.push_back(std::move(c));

        // Sequential pruning and parent updates (keeps functional correctness)
        vector<Node> pruned_next;
        pruned_next.reserve(seq_candidates.size());
        for(auto &cand : seq_candidates){
            const Node &n = cand.first;
            const State &ns = n.s;
            int ng = n.g;
            auto it = best_g.find(ns);
            if(it == best_g.end() || ng < it->second){
                best_g[ns] = ng;
                parent[ns] = cand.second;
                pruned_next.push_back(n);
            }
        }
        vector<Node> next_level_candidates = std::move(pruned_next);

        if (current_level.size() > 0) last_raw_per_node = raw_generated / current_level.size();

        // select top-K using nth_element + sort
        size_t select_size = std::min((size_t)beam_width, next_level_candidates.size());
        if(select_size > 0 && next_level_candidates.size() > select_size){
            std::nth_element(next_level_candidates.begin(),
                             next_level_candidates.begin() + select_size,
                             next_level_candidates.end(),
                             [](const Node &a, const Node &b){ return a.f < b.f; });
        }
        next_level_candidates.resize(select_size);
        std::sort(next_level_candidates.begin(), next_level_candidates.end(), [](const Node &a, const Node &b){ return a.f < b.f; });

        // push selected into pq for next iteration
        for(auto &n : next_level_candidates) pq.push(PQN{n.f, n.g, n.s});

        // check current_level for goal
        for (const auto& cur : current_level) {
            State s = cur.s;
            bool done = (s.boxes & ~target_bits).none();
            if(done) {
                return {cur.g, reconstruct_full_moves(parent, s)};
            }
        }
    }

    return {-1,""};
}

// main
int main(int argc,char** argv){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    if(argc<2){ cerr<<"Usage: "<<argv[0]<<" level.txt\n"; return 1; }
    ifstream fin(argv[1]); if(!fin){ cerr<<"Cannot open "<<argv[1]<<"\n"; return 1; }
    vector<string> lines; string line;
    while(getline(fin,line)) lines.push_back(line);
    if(lines.empty()){ cerr<<"Empty file\n"; return 1; }
    rows=(int)lines.size(); cols=(int)lines[0].size();
    if(rows*cols>=MAX_CELLS){ cerr<<"Map too large\n"; return 1; }
    State start; int idx=0;
    for(int r=0;r<rows;++r){
        if((int)lines[r].size()<cols) lines[r].resize(cols,' ');
        for(int c=0;c<cols;++c,++idx){
            id_to_rc[idx]={r,c};
            char ch=lines[r][c];
            walls[idx]=(ch=='#');
            fragile[idx]=(ch=='@'||ch=='!');
            targets[idx]=(ch=='.'||ch=='O'||ch=='X');
            start.boxes[idx]=(ch=='x'||ch=='X');
            if(ch=='o'||ch=='O'||ch=='!') start.player=idx;
        }
    }
    int N = rows * cols;
    for(int i=0;i<N;++i) if(targets[i]) target_bits.set(i);

    // precompute dead zones
    for(int i=0;i<rows*cols;++i){
        if(walls[i] || targets[i]) continue;
        auto [r,c]=id_to_rc[i];
        bool up=(r==0 || walls[(r-1)*cols+c]), down=(r==rows-1 || walls[(r+1)*cols+c]);
        bool left=(c==0 || walls[r*cols+(c-1)]), right=(c==cols-1 || walls[r*cols+(c+1)]);
        if((up||down)&&(left||right)) dead_zone[i]=true;
    }

    // 2x2 static deadlocks
    auto blocked = [&](int ii) -> bool {
        if (ii < 0 || ii >= N) return true;
        return walls[ii];
    };
    for(int r=1; r<rows-1; ++r) {
        for(int c=1; c<cols-1; ++c) {
            int i00 = r*cols+c, i10 = (r+1)*cols+c, i01 = r*cols+(c+1), i11 = (r+1)*cols+(c+1);
            if (!targets[i00] && !targets[i10] && !targets[i01] && !targets[i11]) {
                if (blocked(i00 - 1) && blocked(i00 - cols)) dead_zone[i00]=true;
                if (blocked(i01 + 1) && blocked(i01 - cols)) dead_zone[i01]=true;
                if (blocked(i10 - 1) && blocked(i10 + cols)) dead_zone[i10]=true;
                if (blocked(i11 + 1) && blocked(i11 + cols)) dead_zone[i11]=true;
            }
        }
    }

    // flood-fill from targets
    bitset<MAX_CELLS> reachable_from_targets;
    queue<int> q;
    for (int i=0; i<N; ++i) if (targets[i]) { reachable_from_targets[i] = 1; q.push(i); }
    while (!q.empty()) {
        int u = q.front(); q.pop();
        auto [r,c] = id_to_rc[u];
        for (int d=0; d<4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr>=0 && nr<rows && nc>=0 && nc<cols) {
                int v = nr*cols + nc;
                if (!walls[v] && !reachable_from_targets[v]) {
                    reachable_from_targets[v] = 1; q.push(v);
                }
            }
        }
    }
    for (int i=0; i<N; ++i) if (!walls[i] && !reachable_from_targets[i]) dead_zone[i] = true;

    // beam widths
    vector<int> beam_widths = {500, 5000, 20000, 100000, 200000, 500000};
    for(int bw : beam_widths){
        auto res = astar_push_solver(start, bw);
        if(res.first >= 0){
            cout << res.second << "\n";
            return 0;
        }
    }
    cout << "No solution found\n";
    return 0;
}
