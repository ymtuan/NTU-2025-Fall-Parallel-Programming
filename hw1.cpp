// Compile: g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1
// Execute: srun -A ACD114118 -n1 -c${threads} ./hw1 ${input}
#include <bits/stdc++.h>
#include <omp.h>
#include <parallel/algorithm>
using namespace std;
const int MAX_CELLS = 256;
const int INF = 1000000007;
struct State {
    int player;
    bitset<MAX_CELLS> boxes;
};
// --- START OPTIMIZATION: Custom Hash and Equality for State ---
struct StateHash {
    size_t operator()(const State& s) const {
        size_t h = std::hash<int>{}(s.player);
       
        const unsigned long long* data = (const unsigned long long*)&s.boxes;
        constexpr int N_WORDS = MAX_CELLS / (sizeof(unsigned long long) * 8);
       
        for (int i = 0; i < N_WORDS; ++i) {
            h ^= std::hash<unsigned long long>{}(data[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};
struct StateEqual {
    bool operator()(const State& a, const State& b) const {
        return a.player == b.player && a.boxes == b.boxes;
    }
};
// --- START OPTIMIZATION: Custom Hash and Equality for Boxes ---
struct BoxesHash {
    size_t operator()(const bitset<MAX_CELLS>& bs) const {
        size_t h = 0;
       
        const unsigned long long* data = (const unsigned long long*)&bs;
        constexpr int N_WORDS = MAX_CELLS / (sizeof(unsigned long long) * 8);
       
        for (int i = 0; i < N_WORDS; ++i) {
            h ^= std::hash<unsigned long long>{}(data[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};
struct BoxesEqual {
    bool operator()(const bitset<MAX_CELLS>& a, const bitset<MAX_CELLS>& b) const {
        return a == b;
    }
};
// --- END OPTIMIZATION ---
int rows=0, cols=0;
vector<bool> walls(MAX_CELLS,false), targets(MAX_CELLS,false), fragile(MAX_CELLS,false), dead_zone(MAX_CELLS,false);
bitset<MAX_CELLS> target_bits;
vector<pair<int,int>> id_to_rc(MAX_CELLS);
int dr[4] = {-1,1,0,0}, dc[4] = {0,0,-1,1};
char dir_char[4] = {'W','S','A','D'}; // up, down, left, right
// ---------- Utilities (removed state_key and parse_state_key) ----------
inline bool in_bounds_idx(int idx) { return idx >=0 && idx < rows*cols; }
// ---------- Deadlock detection (MODIFIED) ----------
// is_deadlock_simple is the original corner check
bool is_deadlock_simple(const State &s) {
    int N = rows*cols;
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
// is_deadlock uses Static Deadlocks + the new, safer Perimeter Check
bool is_deadlock(const State &s) {
    if(is_deadlock_simple(s)) return true;
    int N = rows*cols;
   
    for(size_t i = s.boxes._Find_first(); i < MAX_CELLS; i = s.boxes._Find_next(i)) {
        if(targets[i]) continue;
       
        // Static Dead Zones (original corners + 2x2 precomputed)
        if(dead_zone[i]) return true;
       
        // --- ADDED: Balanced Perimeter Deadlock Check ---
        auto [r,c] = id_to_rc[i];
        // Hard Blockers: Walls or Map Edges (Cannot be moved)
        auto is_wall_blocked = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            return walls[rr * cols + cc];
        };
       
        // Any Blocker: Walls, Map Edges, OR Boxes (Used for the perpendicular axis)
        auto is_blocked_any = [&](int rr, int cc) -> bool {
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return true;
            int idx = rr * cols + cc;
            return walls[idx] || s.boxes[idx];
        };
        // Condition 1: Trapped horizontally by walls AND blocked vertically by ANY obstacle
        // (The box is in a vertical corridor pinned by walls/edges)
        bool wall_block_h = is_wall_blocked(r, c-1) && is_wall_blocked(r, c+1);
        if (wall_block_h) {
            bool blocked_any_v = is_blocked_any(r-1, c) && is_blocked_any(r+1, c);
            if (blocked_any_v) return true;
        }
        // Condition 2: Trapped vertically by walls AND blocked horizontally by ANY obstacle
        // (The box is in a horizontal corridor pinned by walls/edges)
        bool wall_block_v = is_wall_blocked(r-1, c) && is_wall_blocked(r+1, c);
        if (wall_block_v) {
            bool blocked_any_h = is_blocked_any(r, c-1) && is_blocked_any(r, c+1);
            if (blocked_any_h) return true;
        }
    }
    return false;
}
// ---------- Player reachability & BFS player path (unchanged) ----------
bitset<MAX_CELLS> player_reachable(int start, const bitset<MAX_CELLS> &boxes) {
    int N = rows * cols;
    bitset<MAX_CELLS> seen; seen.reset();
    queue<int> q;
    seen[start]=1; q.push(start);
    while(!q.empty()){
        int u = q.front(); q.pop();
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
// ---------- Heuristic (improved with Hungarian assignment) ----------
vector<int> manhattan_target_r, manhattan_target_c;
int heuristic_sum(const State &s){
    int N=rows*cols;
    vector<pair<int,int>> box_positions;
    box_positions.reserve(32);
    for(size_t i = s.boxes._Find_first(); i < MAX_CELLS; i = s.boxes._Find_next(i)) {
        box_positions.push_back(id_to_rc[i]);
    }
    int n = box_positions.size();
    if (n == 0) return 0;
    const int MAX_N = 32;
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
// ---------- Reconstruct moves (MODIFIED) ----------
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
// ---------- Beam A* solver (MODIFIED with adaptive branching estimation) ----------
pair<int,string> astar_push_solver(const State &start, int beam_width, long long initial_branching){
    int N=rows*cols;
    manhattan_target_r.clear(); manhattan_target_c.clear();
    for(int i=0;i<N;++i) if(targets[i]){ manhattan_target_r.push_back(id_to_rc[i].first); manhattan_target_c.push_back(id_to_rc[i].second); }
    struct Node{
        int f,g; State s;
        bool operator<(const Node &o) const{
            if(f!=o.f) return f>o.f; return g<o.g;
        }
    };
    priority_queue<Node> pq;
    unordered_map<State,int,StateHash,StateEqual> best_g;
    unordered_map<State,pair<State,char>,StateHash,StateEqual> parent;
    vector<unordered_map<bitset<MAX_CELLS>, int, BoxesHash, BoxesEqual>> heuristic_caches(omp_get_max_threads());
    int h0 = heuristic_sum(start);
   
    best_g.reserve(beam_width * 20);
    parent.reserve(beam_width * 20);
    best_g[start]=0;
    parent[start] = {State{-1, bitset<MAX_CELLS>{}}, 0};
   
    pq.push(Node{h0,0,start});
    long long last_raw_per_node = 0;
    while(!pq.empty()){
        vector<Node> current_level;
        while(!pq.empty()){
            current_level.push_back(pq.top());
            pq.pop();
        }
       
        // Estimate candidates to decide pruning strategy
        long long est_branching = last_raw_per_node > 0 ? last_raw_per_node : initial_branching;
        long long estimated_candidates = (long long)current_level.size() * est_branching;
        vector<Node> next_level_candidates;
        long long raw_generated = 0;
        // Always use deferred pruning for better performance
        vector<vector<pair<Node, pair<State, char>>>> thread_candidates(omp_get_max_threads());
        #pragma omp parallel for default(none) shared(current_level, N, best_g, thread_candidates, targets, id_to_rc, dr, dc, rows, cols, walls, fragile, dead_zone, dir_char, heuristic_caches, target_bits)
        for(size_t i=0; i<current_level.size(); ++i){
            Node cur=current_level[i];
            State s=cur.s; int g=cur.g;
            int thread_id = omp_get_thread_num();
            // Skip if not current best g (read-only check)
            auto it_g = best_g.find(s);
            if(it_g==best_g.end() || it_g->second != g) continue;
           
            // Check for goal state
            bool done = (s.boxes & ~target_bits).none();
            if(done) {
                continue;
            }
            bitset<MAX_CELLS> reach=player_reachable(s.player, s.boxes);
            for(size_t b = s.boxes._Find_first(); b < MAX_CELLS; b = s.boxes._Find_next(b)){
                auto [br,bc]=id_to_rc[b];
                for(int d=0;d<4;++d){
                    int pr=br-dr[d], pc=bc-dc[d], tr=br+dr[d], tc=bc+dc[d];
                    if(pr<0||pr>=rows||pc<0||pc>=cols) continue;
                    if(tr<0||tr>=rows||tc<0||tc>=cols) continue;
                    int pidx=pr*cols+pc, tidx=tr*cols+tc;
                    if(!reach[pidx]) continue;
                    if(walls[tidx] || s.boxes[tidx] || fragile[tidx] || dead_zone[tidx]) continue;
                    State ns=s; ns.boxes[b]=0; ns.boxes[tidx]=1; ns.player=b;
                    if(is_deadlock(ns)) continue;
                    int ng=g+1;
                    int h;
                    auto &local_cache = heuristic_caches[thread_id];
                    auto hit = local_cache.find(ns.boxes);
                    if (hit != local_cache.end()) {
                        h = hit->second;
                    } else {
                        int computed_h = heuristic_sum(ns);
                        h = computed_h;
                        local_cache[ns.boxes] = h;
                    }
                   
                    // Collect candidate without checking best_g
                    thread_candidates[thread_id].push_back({Node{ng+h, ng, ns}, {s, dir_char[d]}});
                }
            }
        } // End OpenMP parallel for
        // Merge thread-local candidates
        vector<pair<Node, pair<State, char>>> candidates;
        for(auto &vec : thread_candidates){
            candidates.insert(candidates.end(), vec.begin(), vec.end());
        }
        raw_generated = candidates.size();
        // Sequential pruning: update maps only for improving nodes
        vector<Node> pruned_next;
        pruned_next.reserve(candidates.size());
        for(auto &cand : candidates) {
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
        next_level_candidates = std::move(pruned_next);
        if (current_level.size() > 0) {
            last_raw_per_node = raw_generated / current_level.size();
        }
        // Sort and select the beam width (Sequential part with partial_sort)
        size_t select_size = std::min((size_t)beam_width, next_level_candidates.size());
        __gnu_parallel::partial_sort(next_level_candidates.begin(), next_level_candidates.begin() + select_size, next_level_candidates.end(), [](const Node &a, const Node &b){ return a.f < b.f; });
        next_level_candidates.resize(select_size);
       
        // Add to priority queue for the next iteration (Sequential part)
        for(auto &n : next_level_candidates) pq.push(n);
        // Check the current level for the goal state again (as we skipped immediate returns in the parallel loop)
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
// ---------- Main (MODIFIED to compute initial_branching) ----------
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
    // Compute initial valid pushes as proxy for initial_branching
    bitset<MAX_CELLS> reach = player_reachable(start.player, start.boxes);
    long long initial_branching = 0;
    for(size_t b = start.boxes._Find_first(); b < MAX_CELLS; b = start.boxes._Find_next(b)) {
        auto [br, bc] = id_to_rc[b];
        for(int d = 0; d < 4; ++d) {
            int pr = br - dr[d], pc = bc - dc[d], tr = br + dr[d], tc = bc + dc[d];
            if(pr < 0 || pr >= rows || pc < 0 || pc >= cols) continue;
            if(tr < 0 || tr >= rows || tc < 0 || tc >= cols) continue;
            int pidx = pr * cols + pc, tidx = tr * cols + tc;
            if(!reach[pidx]) continue;
            if(walls[tidx] || start.boxes[tidx] || fragile[tidx] || dead_zone[tidx]) continue;
            State ns = start; ns.boxes[b] = 0; ns.boxes[tidx] = 1; ns.player = b;
            if(is_deadlock(ns)) continue;
            ++initial_branching;
        }
    }
    if(initial_branching == 0) initial_branching = 1; // Avoid zero
    // Precompute dead zones (corners with no target)
    for(int i=0;i<rows*cols;++i){
        if(walls[i] || targets[i]) continue;
        auto [r,c]=id_to_rc[i];
        bool up=(r==0 || walls[(r-1)*cols+c]), down=(r==rows-1 || walls[(r+1)*cols+c]);
        bool left=(c==0 || walls[r*cols+(c-1)]), right=(c==cols-1 || walls[r*cols+(c+1)]);
       
        // Original simple corner check
        if((up||down)&&(left||right)) dead_zone[i]=true;
    }
   
    // --- ADDED: 2x2 Static Deadlock Precomputation ---
    auto blocked = [&](int ii) -> bool {
        if (ii < 0 || ii >= N) return true;
        return walls[ii];
    };
    for(int r=1; r<rows-1; ++r) {
        for(int c=1; c<cols-1; ++c) {
            // Check all 4 cells in the 2x2 block
            int i00 = r*cols+c, i10 = (r+1)*cols+c, i01 = r*cols+(c+1), i11 = (r+1)*cols+(c+1);
           
            // Only consider 2x2 blocks of non-target cells
            if (!targets[i00] && !targets[i10] && !targets[i01] && !targets[i11]) {
               
                // Case 1: i00 trapped by Wall Left and Wall Above
                if (blocked(i00 - 1) && blocked(i00 - cols)) dead_zone[i00]=true;
               
                // Case 2: i01 trapped by Wall Right and Wall Above
                if (blocked(i01 + 1) && blocked(i01 - cols)) dead_zone[i01]=true;
                // Case 3: i10 trapped by Wall Left and Wall Below
                if (blocked(i10 - 1) && blocked(i10 + cols)) dead_zone[i10]=true;
                // Case 4: i11 trapped by Wall Right and Wall Below
                if (blocked(i11 + 1) && blocked(i11 + cols)) dead_zone[i11]=true;
            }
        }
    }
    // --- END ADDED ---
   
    // --- ADDED: Flood-fill from targets to mark box-reachable areas ---
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
    // --- END ADDED ---
   
    // Try increasing beam widths
    vector<int> beam_widths = {500, 5000, 100000, 200000};
    // vector<int> beam_widths = {20000, 50000, 80000, 100000};
    for(int bw : beam_widths){
        auto res=astar_push_solver(start, bw, initial_branching);
        if(res.first>=0){
            cout<<res.second<<"\n";
            // cout<<"Solution found with beam width="<<bw<<"\n";
            return 0;
        }
        // cout<<"Failed with beam width="<<bw<<"\n";
    }
    cout<<"No solution found by Beam-A*\n";
    return 0;
}