// Native SynSearch algorithms. See docs/PORTING.md for provenance and semantics.
#include "synsearch.h"
#include "base/abc/abc.h"
#include "base/cmd/cmd.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

ABC_NAMESPACE_IMPL_START
namespace synsearch {
using Seq = std::vector<int>;
struct Action { std::string label, command; };
static const Action catalog[] = {
    {"balance","balance"}, {"rewrite","rewrite"}, {"rewrite-z","rewrite -z"},
    {"refactor","refactor"}, {"refactor-z","refactor -z"}, {"resub","resub"},
    {"resub-z","resub -z"}, {"fraig","fraig"}, {"fx","renode; sop; fx; strash"},
    {"mfs","renode; mfs; strash"}, {"dsd","dsd; strash"}, {"dch","dch; strash"},
    {"extract","extract"}, {"extract-a","extract -a"}, {"collapse","collapse; strash"}
};
struct Config {
    std::string method="mcts", objective="legacy", output, csv;
    int length=24, iterations=64, episodes=50, seed=0;
    int warmup=4, warmepisodes=64, topk=1, rollouts=1, horizon=3, streak=3;
    double cpuct=1, discount=.9, ucb=.4, aw=.5, lw=.2, temperature=-1, mintemp=.001, penalty=-1;
    double alpha=1.5, lambda=1, spike=1.8, levelweight=0, back=.02;
    bool debug=false;
    std::vector<Action> actions;
};
struct Stats { int nodes=0, level=0; double fanout=0; };
static Stats stats(Abc_Ntk_t *n) {
    Stats s; s.nodes=Abc_NtkNodeNum(n); s.level=Abc_NtkLevel(n);
    Abc_Obj_t *obj; int i; double total=0;
    Abc_NtkForEachNode(n,obj,i) total += Abc_ObjFanoutNum(obj);
    // The original parser reads pfan's two-decimal output.
    s.fanout=s.nodes ? std::round(100*total/s.nodes)/100 : 0;
    return s;
}
struct DeleteNetwork { void operator()(Abc_Ntk_t *p) const { if(p) Abc_NtkDelete(p); } };
using Network = std::unique_ptr<Abc_Ntk_t,DeleteNetwork>;
// Candidate commands must not populate the caller's undo stack or mutate its
// frame state. ABC's explicit frame API scopes all temporary evaluations.
class WorkingFrame {
    Abc_Frame_t *previous;
    bool entered=true;
public:
    Abc_Frame_t *frame;
    WorkingFrame():frame(Abc_FrameCreate()) { previous=Abc_FrameEnter(frame); }
    void leave() { if(entered){Abc_FrameLeave(previous);entered=false;} }
    ~WorkingFrame() { leave();Abc_FrameDestroy(frame); }
    WorkingFrame(const WorkingFrame&)=delete;
    WorkingFrame& operator=(const WorkingFrame&)=delete;
};
struct Entry { Network net; Stats s; bool valid=false; };
static double signedGain(int old, int now, double base) {
    double r=std::sqrt(std::abs(double(old)-now)/base); return old>now ? r : -r;
}
static double reward(Stats a, Stats b, double an, double ln, const Config &c) {
    if(c.objective=="nodes")return signedGain(a.nodes,b.nodes,an);
    if(c.objective=="level")return signedGain(a.level,b.level,ln);
    return (5*signedGain(a.nodes,b.nodes,an)+2*signedGain(a.level,b.level,ln))/7;
}
static double cost(Stats base, Stats s, double aw=5, double lw=2) {
    return (aw*double(s.nodes)/std::max(base.nodes,1)+lw*double(s.level)/std::max(base.level,1))/(aw+lw);
}
static bool better(Stats a,Stats b,const Config &c) {
    if(c.objective=="level")return a.level<b.level || (a.level==b.level && a.nodes<b.nodes);
    return a.nodes<b.nodes || (a.nodes==b.nodes && a.level<b.level);
}
static double branchCost(Stats base,Stats s,const Config &c) {
    return c.objective=="legacy" ? cost(base,s) : cost(base,s,c.aw,c.lw);
}
static std::string quote(const std::string &s) {
    std::ostringstream o; o << '"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\') o << '\\' << c;
        else if(c<32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << c;
    } return o.str()+'"';
}
class Engine {
public:
    const Config &c; Abc_Frame_t *frame; std::map<Seq,Entry> cache;
    Stats initial; double an=1,ln=1; size_t evaluations=0,hits=0;
    bool failed=false; std::ostringstream trace;
    explicit Engine(const Config &cfg,Abc_Frame_t *f,Abc_Ntk_t *root):c(cfg),frame(f) {
        Entry e; e.net.reset(Abc_NtkDup(root)); e.s=stats(root); e.valid=true;
        initial=e.s; cache.emplace(Seq{},std::move(e));
    }
    Entry &eval(const Seq &p) {
        auto it=cache.find(p); if(it!=cache.end()) { ++hits; return it->second; }
        Seq parent(p.begin(),p.end()-1); Entry &prev=eval(parent); Entry next;
        if(prev.valid) {
            Abc_FrameReplaceCurrentNetwork(frame,Abc_NtkDup(prev.net.get()));
            ++evaluations;
            int rc=Cmd_CommandExecute(frame,c.actions[p.back()].command.c_str());
            Abc_Ntk_t *n=Abc_FrameReadNtk(frame);
            if(rc==0 && n && Abc_NtkIsStrash(n)) {
                next.net.reset(Abc_NtkDup(n)); next.s=stats(n); next.valid=true;
            }
        }
        return cache.emplace(p,std::move(next)).first->second;
    }
    void baseline() {
        Abc_FrameReplaceCurrentNetwork(frame,Abc_NtkDup(cache.at({}).net.get()));
        if(Cmd_CommandExecute(frame,"balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance")) { failed=true; return; }
        Stats h=stats(Abc_FrameReadNtk(frame));
        an=initial.nodes>h.nodes ? initial.nodes-h.nodes : std::max(initial.nodes/10.,1.);
        ln=initial.level>h.level ? initial.level-h.level : std::max(initial.level/10.,1.);
    }
    bool lexless(const Seq &a,const Seq &b) const {
        return std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end(),[&](int x,int y){return c.actions[x].label<c.actions[y].label;});
    }
    void event(const char *kind,int step,const Seq &p,double value) {
        if(!c.debug) return;
        if(trace.tellp()>0) trace << ',';
        trace << "{\"event\":" << quote(kind) << ",\"step\":" << step << ",\"sequence\":" << sequence(p)
              << ",\"value\":" << std::setprecision(17) << value << '}';
    }
    std::string sequence(const Seq &p,bool commands=false) const {
        std::ostringstream o; o << '['; for(size_t i=0;i<p.size();++i) { if(i)o<<','; o<<quote(commands?c.actions[p[i]].command:c.actions[p[i]].label); } return o.str()+']';
    }
};
// CPython-compatible MT19937 integer seeding and draws for the reference
// algorithms using random.Random. NumPy LinUCB tie draws are documented separately.
class PythonRandom {
    std::array<unsigned,624> state{}; size_t index=624;
public:
    explicit PythonRandom(unsigned seed) {
        state[0]=19650218U;
        for(unsigned i=1;i<624;++i)state[i]=1812433253U*(state[i-1]^(state[i-1]>>30))+i;
        unsigned i=1;
        for(unsigned k=0;k<624;++k){state[i]=(state[i]^((state[i-1]^(state[i-1]>>30))*1664525U))+seed;if(++i>=624){state[0]=state[623];i=1;}}
        for(unsigned k=0;k<623;++k){state[i]=(state[i]^((state[i-1]^(state[i-1]>>30))*1566083941U))-i;if(++i>=624){state[0]=state[623];i=1;}}
        state[0]=0x80000000U;
    }
    unsigned draw() {
        if(index>=624){for(size_t i=0;i<624;++i){unsigned y=(state[i]&0x80000000U)|(state[(i+1)%624]&0x7fffffffU);state[i]=state[(i+397)%624]^(y>>1)^((y&1)?0x9908b0dfU:0);}index=0;}
        unsigned y=state[index++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;y^=(y<<15)&0xefc60000U;y^=y>>18;return y;
    }
    double uniform(){unsigned a=draw()>>5,b=draw()>>6;return (a*67108864.0+b)/9007199254740992.0;}
    int below(int n){unsigned bits=0,t=unsigned(n);while(t){++bits;t>>=1;}unsigned v;do{v=draw()>>(32-bits);}while(v>=unsigned(n));return int(v);}
};
struct Bandit {
    std::vector<int> counts; std::vector<double> q; int total=0; PythonRandom rng;
    Bandit(int n,int seed):counts(n),q(n),rng(seed){}
    int select(double c) {
        std::vector<int> ties; double best=-1e100;
        for(size_t i=0;i<counts.size();++i) if(!counts[i]) ties.push_back(int(i));
        if(ties.empty()) for(size_t i=0;i<counts.size();++i) {
            double s=q[i]+c*std::sqrt(std::log(std::max(total,1))/counts[i]);
            if(s>best){best=s;ties.clear();} if(s==best)ties.push_back(int(i));
        }
        int a=ties[rng.below(int(ties.size()))]; ++counts[a];++total;return a;
    }
    void update(int a,double r) { q[a]+=(r-q[a])/counts[a]; }
};
struct Node;
struct Edge { int action,visits=0; double r=0; std::unique_ptr<Node> child; };
struct Node { Seq prefix; Stats s; int visits=0; double value=0; std::vector<Edge> edges; };
static void recompute(Node &n,double discount) {
    n.value=0; if(n.edges.empty())return;
    double best=-1e100; for(auto &e:n.edges)best=std::max(best,e.r+e.child->value); n.value=discount*best;
}
static void expand(Engine &e,Node &n) {
    if(!n.edges.empty()||int(n.prefix.size())>=e.c.length)return;
    for(size_t a=0;a<e.c.actions.size();++a) {
        Seq p=n.prefix;p.push_back(int(a));Entry &v=e.eval(p);
        if(!v.valid){e.failed=true;return;}
        Edge edge;edge.action=int(a);edge.r=reward(n.s,v.s,e.an,e.ln,e.c);
        edge.child.reset(new Node);edge.child->prefix=p;edge.child->s=v.s;n.edges.push_back(std::move(edge));
    } recompute(n,e.c.discount);
}
static Seq mcts(Engine &e,const Seq &prefix) {
    std::unique_ptr<Node> root(new Node);root->prefix=prefix;root->s=e.eval(prefix).s;
    while(int(root->prefix.size())<e.c.length && !e.failed) {
        for(int it=0;it<e.c.iterations && !e.failed;++it) {
            Node *n=root.get();std::vector<std::pair<Node*,Edge*>> path;
            while(int(n->prefix.size())<e.c.length) {
                if(n->edges.empty()){expand(e,*n);break;}
                Edge *selected=nullptr;double best=-1e100;
                for(auto &edge:n->edges) {
                    double score=edge.r+edge.child->value+e.c.cpuct/e.c.actions.size()*std::sqrt(std::max(n->visits,1))/(edge.visits+1);
                    if(!selected||score>best||(score==best && e.c.actions[edge.action].label>e.c.actions[selected->action].label)){best=score;selected=&edge;}
                }
                path.emplace_back(n,selected);n=selected->child.get();
            }
            if(!path.empty()) {++path[0].first->visits;for(auto &v:path){++v.second->visits;++v.second->child->visits;}}
            for(auto i=path.rbegin();i!=path.rend();++i){recompute(*i->second->child,e.c.discount);recompute(*i->first,e.c.discount);}
            recompute(*root,e.c.discount); e.event("mcts_iteration",it+1,n->prefix,root->value);
        }
        if(e.failed||root->edges.empty())break;
        Edge *best=nullptr;
        for(auto &v:root->edges)if(!best||v.r+v.child->value>best->r+best->child->value||
            (v.r+v.child->value==best->r+best->child->value && e.c.actions[v.action].label>e.c.actions[best->action].label))best=&v;
        e.event("mcts_step",int(root->prefix.size())+1,best->child->prefix,best->r+best->child->value);
        auto next=std::move(best->child);root=std::move(next);
    }return root->prefix;
}
static double banditReward(Engine &e,Stats a,Stats b) {
    return (e.c.aw*signedGain(a.nodes,b.nodes,std::max(a.nodes,1))+e.c.lw*signedGain(a.level,b.level,std::max(a.level,1)))/(e.c.aw+e.c.lw);
}
static Seq ucb(Engine &e,bool prefix) {
    std::vector<Bandit> bs;for(int i=0;i<(prefix?e.c.length:1);++i)bs.emplace_back(int(e.c.actions.size()),e.c.seed+i);
    Seq best; double bestcost=1;
    for(int ep=0;ep<e.c.episodes;++ep) {
        Seq p;Stats prev=e.initial;bool valid=true;
        for(int step=0;step<e.c.length;++step) {
            Bandit &b=bs[prefix?step:0];int a=b.select(e.c.ucb);p.push_back(a);
            if(prefix) {
                Entry &v=e.eval(p);double r=(!v.valid||v.s.nodes>int(prev.nodes*e.c.spike))?e.c.penalty:banditReward(e,prev,v.s);
                b.update(a,r);e.event("ucb_step",step+1,p,r);if(!v.valid){valid=false;break;}prev=v.s;
            }
        }
        Entry &v=e.eval(p);
        if(!prefix) {
            double r=(!v.valid||v.s.nodes>int(e.initial.nodes*e.c.spike))?e.c.penalty:banditReward(e,e.initial,v.s);
            for(int a:std::set<int>(p.begin(),p.end()))bs[0].update(a,r);
            valid=v.valid && !(e.initial.nodes>0 && v.s.nodes>int(e.initial.nodes*e.c.spike));
        }
        if(valid && v.valid && cost(e.initial,v.s,e.c.aw,e.c.lw)<bestcost){best=p;bestcost=cost(e.initial,v.s,e.c.aw,e.c.lw);}
        e.event("ucb_episode",ep+1,best,bestcost);
    }return best;
}
static Seq hybrid(Engine &e) {
    int steps=std::min(e.c.warmup,e.c.length);if(!steps)return mcts(e,{});
    std::vector<Bandit> bs;for(int i=0;i<steps;++i)bs.emplace_back(int(e.c.actions.size()),e.c.seed+i);
    std::vector<Seq> candidates;
    auto less=[&](const Seq&a,const Seq&b){double x=branchCost(e.initial,e.eval(a).s,e.c),y=branchCost(e.initial,e.eval(b).s,e.c);return x!=y?x<y:e.lexless(a,b);};
    for(int ep=0;ep<e.c.warmepisodes;++ep) {
        Seq p;Stats prev=e.initial;bool ok=true;
        for(int s=0;s<steps;++s){int a=bs[s].select(e.c.ucb);p.push_back(a);Entry &v=e.eval(p);double r=v.valid?reward(prev,v.s,e.an,e.ln,e.c):-1;bs[s].update(a,r);if(!v.valid){ok=false;break;}prev=v.s;}
        if(ok && std::find(candidates.begin(),candidates.end(),p)==candidates.end()){candidates.push_back(p);std::sort(candidates.begin(),candidates.end(),less);if(int(candidates.size())>e.c.topk)candidates.pop_back();}
        e.event("warmup_episode",ep+1,p,ok?branchCost(e.initial,prev,e.c):-1);
    }
    if(candidates.empty())candidates.push_back({});
    Seq best;bool first=true;
    for(const auto &p:candidates){Seq result=mcts(e,p);if(first||less(result,best)){best=result;first=false;}}
    return best;
}
static double uniform(PythonRandom &r){return r.uniform();}
static int randint(PythonRandom &r,int n){return r.below(n);}
static int randint(std::mt19937 &r,int n){return std::uniform_int_distribution<int>(0,n-1)(r);}
static Seq neighbor(Seq p,const Config &c,PythonRandom &r) {
    if(p.empty())return p;
    double move=uniform(r);int n=int(p.size()),m=int(c.actions.size());
    if(move<.6||n==1){int i=randint(r,n);std::vector<int> choices;for(int a=0;a<m;++a)if(a!=p[i])choices.push_back(a);if(!choices.empty())p[i]=choices[randint(r,int(choices.size()))];}
    else if(move<.8){int i=randint(r,n),j; if(n<=21){j=randint(r,n-1);if(j==i)j=n-1;}else{do{j=randint(r,n);}while(j==i);}std::swap(p[i],p[j]);}
    else{int size=2+randint(r,std::min(4,n)-1),start=randint(r,n-size+1);for(int i=0;i<size;++i)p[start+i]=randint(r,m);}
    return p;
}
static Seq anneal(Engine &e) {
    const char *seed[]={"balance","rewrite","refactor","balance","rewrite","rewrite-z","balance","refactor-z","rewrite-z","balance"};
    Seq pattern,p;for(auto s:seed)for(size_t a=0;a<e.c.actions.size();++a)if(e.c.actions[a].label==s)pattern.push_back(int(a));
    if(pattern.empty())for(size_t a=0;a<e.c.actions.size();++a)pattern.push_back(int(a));
    for(int i=0;i<e.c.length;++i)p.push_back(pattern[i%pattern.size()]);
    auto energy=[&](const Seq &s){Entry &v=e.eval(s);if(!v.valid){e.failed=true;return 1e100;}return cost(e.initial,v.s,e.c.aw,e.c.lw);};
    PythonRandom rng(e.c.seed);double cur=energy(p),bestenergy=cur,temp=e.c.temperature;Seq best=p;
    if(temp<0){double total=0;int count=0;for(int i=0;i<32;++i){double d=energy(neighbor(p,e.c,rng))-cur;if(d>0){total+=d;++count;}}temp=count?-total/count/std::log(.8):.05;}
    temp=std::max(temp,e.c.mintemp);
    for(int it=0;it<e.c.iterations&&!e.failed;++it){double t=temp*std::pow(e.c.mintemp/temp,double(it)/std::max(e.c.iterations-1,1));Seq next=neighbor(p,e.c,rng);double value=energy(next),delta=value-cur;if(delta<=0||uniform(rng)<std::exp(-delta/std::max(t,1e-12))){p=next;cur=value;}if(cur<bestenergy){best=p;bestenergy=cur;}e.event("sa_iteration",it+1,p,cur);}
    return best;
}
using Context=std::array<double,10>;
struct LinearArm {
    std::array<Context,10> inverse{};Context b{};
    explicit LinearArm(double lambda){for(int i=0;i<10;++i)inverse[i][i]=1/lambda;}
    Context product(const Context &x)const{Context z{};for(int i=0;i<10;++i)for(int j=0;j<10;++j)z[i]+=inverse[i][j]*x[j];return z;}
    static double dot(const Context &x,const Context &y){double v=0;for(int i=0;i<10;++i)v+=x[i]*y[i];return v;}
    double score(const Context &x,double alpha)const{Context z=product(x);return dot(z,b)+alpha*std::sqrt(std::max(0.,dot(z,x)));}
    void update(const Context &x,double r){Context z=product(x);double d=1+dot(z,x);for(int i=0;i<10;++i){b[i]+=r*x[i];for(int j=0;j<10;++j)inverse[i][j]-=z[i]*z[j]/d;}}
};
struct DepthBest {Stats s;int origin;};
static Seq linucb(Engine &e) {
    const Config &c=e.c;int n=int(c.actions.size()),h=std::min(c.horizon,c.length),tail=std::max(c.length-h,0);
    std::vector<std::vector<LinearArm>> policies(c.length,std::vector<LinearArm>(n,LinearArm(c.lambda)));
    std::vector<std::mt19937> rngs;for(int s=0;s<c.length;++s)rngs.emplace_back(c.seed+s);
    PythonRandom rollout(c.seed);std::vector<std::set<int>> excluded(c.length);std::vector<bool> backtracked(c.length,false);
    std::map<int,DepthBest> depthbest;Seq current;
    while(int(current.size())<c.length) {
        int step=int(current.size());Stats base=e.eval(current).s;
        std::vector<Stats> next(n);std::vector<bool> valid(n,false);bool any=false;
        for(int a=0;a<n;++a){if(excluded[step].count(a))continue;Seq p=current;p.push_back(a);Entry &v=e.eval(p);valid[a]=v.valid&&v.s.nodes<=int(std::max(base.nodes,1)*c.spike);next[a]=v.s;any=any||valid[a];}
        if(!any)break;
        int selected=-1,last=-1,streak=0;Seq tailbest;Stats tailstats;bool hastail=false;
        for(int it=0;it<c.episodes;++it) {
            std::vector<Context> contexts(n);std::vector<double> scores(n,-1e100);
            for(int a=0;a<n;++a)if(valid[a]) {
                Context &x=contexts[a];x[0]=1;x[1]=double(next[a].nodes)/std::max(e.initial.nodes,1);x[2]=double(next[a].level)/std::max(e.initial.level,1);x[3]=next[a].fanout/std::max(e.initial.fanout,1.);
                std::vector<double> ng,lg;
                for(int k=0;k<c.rollouts && h>0;++k) {
                    Seq p=current;std::vector<Stats> samples;bool ok=true;
                    for(int d=0;d<h;++d){p.push_back(d?randint(rollout,n):a);Entry &v=e.eval(p);if(!v.valid){ok=false;break;}samples.push_back(v.s);}
                    if(!ok)continue;
                    for(int d=0;d<h;++d){auto found=depthbest.find(step+d+1);if(found==depthbest.end()||better(samples[d],found->second.s,c))depthbest[step+d+1]={samples[d],step};}
                    Stats end=samples.back();ng.push_back(double(std::max(base.nodes,1)-end.nodes)/std::max(base.nodes,1));lg.push_back(double(std::max(base.level,1)-end.level)/std::max(base.level,1));
                    if(step==tail && (!hastail||better(end,tailstats,c))){hastail=true;tailbest=p;tailstats=end;}
                }
                auto moments=[&](const std::vector<double>&v,int off){if(v.empty())return;double sum=0,variance=0;for(double z:v)sum+=z;double mean=sum/v.size();for(double z:v)variance+=(z-mean)*(z-mean);x[off]=mean;x[off+1]=*std::max_element(v.begin(),v.end());x[off+2]=std::sqrt(variance/v.size());};
                moments(ng,4);moments(lg,7);scores[a]=policies[step][a].score(x,c.alpha);
            }
            double best=*std::max_element(scores.begin(),scores.end());std::vector<int> ties;
            for(int a=0;a<n;++a)if(valid[a]&&std::abs(scores[a]-best)<=1e-8+1e-5*std::abs(best))ties.push_back(a);
            selected=ties[randint(rngs[step],int(ties.size()))];
            double r=(1-c.levelweight)*double(base.nodes-next[selected].nodes)/std::max(base.nodes,1)+c.levelweight*double(base.level-next[selected].level)/std::max(base.level,1);
            policies[step][selected].update(contexts[selected],r);
            streak=selected==last?streak+1:1;last=selected;e.event("linucb_iteration",step+1,current,r);
            if(streak>=c.streak)break;
        }
        if(step==tail){if(hastail)current=tailbest;break;}
        if(selected<0)break;
        current.push_back(selected);Stats now=e.eval(current).s;e.event("linucb_step",step+1,current,now.nodes);
        auto found=depthbest.find(step+1);
        if(found!=depthbest.end()) {
            int target=found->second.origin;double gap=c.objective=="level"
                ? std::max(double(now.level-found->second.s.level)/std::max(e.initial.level,1),0.)
                : std::max(double(now.nodes-found->second.s.nodes)/std::max(e.initial.nodes,1),0.);
            if(gap>c.back && target<step && !backtracked[target]) {
                backtracked[target]=true;excluded[target].insert(current[target]);current.resize(target);
                for(int s=target;s<c.length;++s){policies[s]=std::vector<LinearArm>(n,LinearArm(c.lambda));rngs[s].seed(c.seed+s);}
                depthbest.erase(depthbest.upper_bound(target),depthbest.end());e.event("linucb_backtrack",target+1,current,gap);
            }
        }
    }return current;
}
static void help() {
    Abc_Print(1,"Native SynSearch: search the current strashed network in memory.\n"
      "usage: synsearch [--algorithm mcts|hybrid|sa|ucb1|ucb1-prefix|linucb] [options]\n"
      "  --objective legacy|nodes|level (default legacy; explicit objective overrides reward weights)\n"
      "  --sequence-length N / --steps N  --search-iterations N  --episodes N\n"
      "  --seed N  --actions balance,rewrite,rewrite-z,...  --debug-search\n"
      "  --cpuct F  --mu-discount F  --ucb-c F  --spike-factor F  --severe-negative-reward F\n"
      "  --warmup-steps N  --warmup-episodes N  --warmup-top-k N\n"
      "  --and-weight F  --lev-weight F  --initial-temperature F  --min-temperature F\n"
      "  --alpha F  --lambda F  --long-term-rollouts N  --long-term-horizon N\n"
      "  --early-stop-same-arm-streak N  --level-reward-weight F  --return-back-threshold F\n"
      "  --output result.json  --csv summary.csv (append)\n"
      "Search replaces the current network only on success. See docs/PORTING.md.\n");
}
static bool number(const std::string&s,double &out){char *end=nullptr;errno=0;out=std::strtod(s.c_str(),&end);return !errno&&end!=s.c_str()&&!*end&&std::isfinite(out);}
static bool parse(int argc,char **argv,Config &c) {
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;++i){std::string k=argv[i];if(k=="--debug-search"){c.debug=true;continue;}if(i+1>=argc){Abc_Print(-1,"Missing value for %s.\n",k.c_str());return false;}if(args.count(k)){Abc_Print(-1,"Duplicate option %s.\n",k.c_str());return false;}args[k]=argv[++i];}
    if(args.count("--algorithm")){c.method=args["--algorithm"];args.erase("--algorithm");}
    if(c.method=="alphasyn")c.method="mcts";
    if(c.method=="hybridsyn")c.method="hybrid";
    if(c.method=="sasyn")c.method="sa";
    bool bandit=c.method=="ucb1"||c.method=="ucb1-prefix"||c.method=="linucb";
    if(!bandit&&c.method!="mcts"&&c.method!="hybrid"&&c.method!="sa"){Abc_Print(-1,"Unknown algorithm.\n");return false;}
    if(bandit)c.length=10;
    if(c.method=="linucb")c.episodes=150;
    if(c.method=="sa"){c.iterations=1000;c.aw=.7;c.lw=.3;}
    if(c.method=="ucb1")c.ucb=2;
    std::map<std::string,int*> ints={{"--sequence-length",&c.length},{"--steps",&c.length},{"--search-iterations",&c.iterations},{"--episodes",&c.episodes},{"--seed",&c.seed},{"--warmup-steps",&c.warmup},{"--warmup-episodes",&c.warmepisodes},{"--warmup-top-k",&c.topk},{"--long-term-rollouts",&c.rollouts},{"--long-term-horizon",&c.horizon},{"--early-stop-same-arm-streak",&c.streak}};
    std::map<std::string,double*> floats={{"--severe-negative-reward",&c.penalty},{"--cpuct",&c.cpuct},{"--mu-discount",&c.discount},{"--ucb-c",&c.ucb},{"--and-weight",&c.aw},{"--lev-weight",&c.lw},{"--initial-temperature",&c.temperature},{"--min-temperature",&c.mintemp},{"--alpha",&c.alpha},{"--lambda",&c.lambda},{"--spike-factor",&c.spike},{"--level-reward-weight",&c.levelweight},{"--return-back-threshold",&c.back}};
    if(args.count("--steps")&&args.count("--sequence-length")){Abc_Print(-1,"Use only one length option.\n");return false;}
    for(const auto &kv:args){double v;
        if(ints.count(kv.first)){if(!number(kv.second,v)||v<0||v>1000000||v!=std::floor(v))return false;*ints[kv.first]=int(v);}
        else if(floats.count(kv.first)){if(!number(kv.second,v)||(v<0&&kv.first!="--severe-negative-reward"))return false;*floats[kv.first]=v;}
        else if(kv.first=="--objective")c.objective=kv.second;
        else if(kv.first=="--output")c.output=kv.second;
        else if(kv.first=="--csv")c.csv=kv.second;
        else if(kv.first=="--actions") {
            std::istringstream in(kv.second);std::string label;std::set<std::string> seen;
            while(std::getline(in,label,',')){bool found=false;for(const auto &a:catalog)if(label==a.label||label==a.command){if(!seen.insert(a.label).second)return false;c.actions.push_back(a);found=true;break;}if(!found){Abc_Print(-1,"Unknown action: %s. Use action labels from the README.\n",label.c_str());return false;}}
            if(c.actions.empty())return false;
        }else{Abc_Print(-1,"Unknown option: %s.\n",kv.first.c_str());return false;}
    }
    if(c.objective!="legacy"&&c.objective!="nodes"&&c.objective!="level")return false;
    // Explicit single-metric objectives apply to all policy decisions. Legacy
    // remains byte-for-byte compatible with the previous objective semantics.
    if(c.objective!="legacy") {
        c.aw=c.objective=="nodes"?1:0;c.lw=c.objective=="level"?1:0;
        c.levelweight=c.objective=="level"?1:0;
    }
    if(c.actions.empty()){if(bandit){for(int a:{1,2,3,4,0,5,6})c.actions.push_back(catalog[a]);}else c.actions.assign(catalog,catalog+7);}
    return c.iterations>0&&c.episodes>0&&c.warmepisodes>0&&c.topk>0&&c.streak>0&&c.lambda>0&&c.mintemp>0&&c.aw+c.lw>0&&c.discount<=1&&c.levelweight<=1&&c.spike>0&& (c.method!="linucb"|| (c.rollouts>0&&c.horizon>0));
}
static int command(Abc_Frame_t *frame,int argc,char **argv) {
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--help"||std::string(argv[i])=="-h"){help();return 0;}
    Config c;if(!parse(argc,argv,c)){Abc_Print(-1,"Invalid synsearch options. Use synsearch --help.\n");return 1;}
    Abc_Ntk_t *input=Abc_FrameReadNtk(frame);
    if(!input||!Abc_NtkIsStrash(input)){Abc_Print(-1,"Read a network and run strash first.\n");return 1;}
    // Validate output destinations before spending the search budget.
    std::ofstream json,csv;
    if(!c.output.empty()){json.open(c.output);if(!json){Abc_Print(-1,"Cannot open result file.\n");return 1;}}
    bool csvheader=false;
    if(!c.csv.empty()){std::ifstream prior(c.csv);csvheader=prior.peek()==std::ifstream::traits_type::eof();csv.open(c.csv,std::ios::app);if(!csv){Abc_Print(-1,"Cannot open CSV file.\n");return 1;}}
    std::string name=Abc_NtkName(input);
    WorkingFrame workspace;
    auto start=std::chrono::steady_clock::now();Seq result;
    Engine e(c,workspace.frame,input);
    if(c.method=="mcts"||c.method=="hybrid")e.baseline();
    if(!e.failed){if(c.method=="mcts")result=mcts(e,{});else if(c.method=="hybrid")result=hybrid(e);else if(c.method=="sa")result=anneal(e);else if(c.method=="linucb")result=linucb(e);else result=ucb(e,c.method=="ucb1-prefix");}
    Entry &final=e.eval(result);
    if(e.failed||!final.valid){Abc_Print(-1,"Search evaluation failed; original network restored.\n");return 1;}
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ostringstream recipe;for(size_t i=0;i<result.size();++i){if(i)recipe<<"; ";recipe<<c.actions[result[i]].command;}
    if(json.is_open()) {
        json<<std::setprecision(17)<<"{\"schema_version\":1,\"implementation\":\"native C++ SynSearch\",\"algorithm\":"<<quote(c.method)<<",\"objective\":"<<quote(c.objective)<<",\"design_name\":"<<quote(name)<<",\"seed\":"<<c.seed
            <<",\"sequence\":"<<e.sequence(result)<<",\"commands\":"<<e.sequence(result,true)<<",\"recipe\":"<<quote(recipe.str())
            <<",\"initial_and\":"<<e.initial.nodes<<",\"initial_lev\":"<<e.initial.level<<",\"final_and\":"<<final.s.nodes<<",\"final_lev\":"<<final.s.level
            <<",\"total_runtime_sec\":"<<seconds<<",\"evaluations\":"<<e.evaluations<<",\"cache_hits\":"<<e.hits
            <<",\"config\":{\"sequence_length\":"<<c.length<<",\"search_iterations\":"<<c.iterations<<",\"episodes\":"<<c.episodes<<",\"cpuct\":"<<c.cpuct<<",\"mu_discount\":"<<c.discount<<",\"ucb_c\":"<<c.ucb
            <<",\"warmup_steps\":"<<c.warmup<<",\"warmup_episodes\":"<<c.warmepisodes<<",\"warmup_top_k\":"<<c.topk
            <<",\"and_weight\":"<<c.aw<<",\"lev_weight\":"<<c.lw<<",\"initial_temperature\":"<<c.temperature<<",\"min_temperature\":"<<c.mintemp
            <<",\"alpha\":"<<c.alpha<<",\"lambda\":"<<c.lambda<<",\"long_term_rollouts\":"<<c.rollouts<<",\"long_term_horizon\":"<<c.horizon
            <<",\"early_stop_same_arm_streak\":"<<c.streak<<",\"level_reward_weight\":"<<c.levelweight<<",\"return_back_threshold\":"<<c.back<<",\"spike_factor\":"<<c.spike<<",\"severe_negative_reward\":"<<c.penalty<<",\"actions\":[";
        for(size_t i=0;i<c.actions.size();++i){if(i)json<<',';json<<quote(c.actions[i].label);}json<<"]},\"trace\":["<<e.trace.str()<<"]}\n";json.flush();
        if(!json){Abc_Print(-1,"Result write failed; original network restored.\n");return 1;}
    }
    if(csv.is_open()){
        auto escape=[](const std::string &s){std::string r="\"";for(char x:s){r+=x;if(x=='"')r+='"';}return r+'"';};
        if(csvheader)csv<<"algorithm,design_name,seed,initial_and,initial_lev,final_and,final_lev,total_runtime_sec,recipe\n";
        csv<<c.method<<','<<escape(name)<<','<<c.seed<<','<<e.initial.nodes<<','<<e.initial.level<<','<<final.s.nodes<<','<<final.s.level<<','<<seconds<<','<<escape(recipe.str())<<'\n';csv.flush();
        if(!csv){Abc_Print(-1,"CSV write failed; original network restored.\n");return 1;}
    }
    workspace.leave();
    Abc_FrameSetCurrentNetwork(frame,Abc_NtkDup(final.net.get()));
    Abc_Print(1,"SynSearch %s: and %d -> %d, lev %d -> %d, %.3f sec, %llu evaluations\nRecipe: %s\n",c.method.c_str(),e.initial.nodes,final.s.nodes,e.initial.level,final.s.level,seconds,(unsigned long long)e.evaluations,recipe.str().c_str());
    return 0;
}
} // namespace synsearch
ABC_NAMESPACE_IMPL_END
ABC_NAMESPACE_HEADER_START
void SynSearch_Init(Abc_Frame_t *frame) { Cmd_CommandAdd(frame,"Synthesis","synsearch",synsearch::command,1); }
ABC_NAMESPACE_HEADER_END
