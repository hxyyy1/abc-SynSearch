#!/usr/bin/env python3
"""Fixed-length policy-parameter Cartesian grid search on VTR.

Scan complete grids on or1200; confirm top three on fir (seed 0). Freeze one configuration per algorithm and
objective, then evaluate the four other circuits and unused seeds. Reuses the
previous experiment as an explicit control. MCTS is deterministic: seed 0 only.
ABC's algorithm implementation is unchanged. Independent processes are pinned
to distinct physical cores; their network caches are never shared.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from statistics import mean
from vtr_objectives import METHODS, ACTIONS, sha, save, stats, resources, cec

GIB=1024**3
DESIGNS=('bfly','dscg','fir','ode','or1200','syn2')
TRAIN=('or1200','fir')
HOLDOUT=('bfly','dscg','ode','syn2')


def configurations(method):
    """Full Cartesian grids of policy parameters; length and budget stay fixed."""
    from itertools import product
    base={'steps':10,'search-iterations':96 if method=='sa' else 12,
          'episodes':8 if method=='linucb' else 24,'warmup-steps':3,
          'warmup-episodes':16,'warmup-top-k':1,'long-term-rollouts':1,
          'long-term-horizon':3,'early-stop-same-arm-streak':3}
    axes={
        'mcts': [('cpuct',[.2,1.,3.]),('mu-discount',[.7,.9,1.])],
        'hybrid': [('cpuct',[.2,1.,3.]),('ucb-c',[.1,.4,1.])],
        'sa': [('initial-temperature',[-1.,.005,.02,.1]),('min-temperature',[.0001,.001])],
        'ucb1': [('ucb-c',[0.,.01,.05,.1,.4,1.,2.,4.])],
        'ucb1-prefix': [('ucb-c',[0.,.01,.05,.1,.4,1.,2.,4.])],
        'linucb': [('alpha',[.25,1.5,3.]),('lambda',[.1,1.,10.])]
    }[method]
    defaults={'cpuct':1.,'mu-discount':.9,'ucb-c':2. if method=='ucb1' else .4,
              'initial-temperature':-1.,'min-temperature':.001,'alpha':1.5,'lambda':1.}
    baseline=dict(base,**{k:defaults[k] for k,_ in axes})
    result={'baseline':baseline}
    for vals in product(*(values for _,values in axes)):
        delta=dict(zip((k for k,_ in axes),vals))
        if all(v==defaults[k] for k,v in delta.items()):continue
        name='_'.join(k+'-'+format(v,'g').replace('.','p').replace('-','m') for k,v in delta.items())
        result[name]=dict(base,**delta)
    return result


def key(job):
    return (job['objective'],job['algorithm'],job['config_id'],job['design'],job['seed'])


def physical_cpus(limit):
    rows=subprocess.check_output(['lscpu','-p=CPU,CORE'],text=True)
    allowed=os.sched_getaffinity(0);seen=set();cpus=[]
    for line in rows.splitlines():
        if line.startswith('#'):continue
        cpu,core=map(int,line.split(','))
        if cpu in allowed and core not in seen:seen.add(core);cpus.append(cpu)
    return cpus[:limit]


class Memory:
    def __init__(self):
        group=subprocess.check_output(['systemctl','show',f'user-{os.getuid()}.slice','-p','ControlGroup','--value'],text=True).strip()
        self.path=Path('/sys/fs/cgroup')/group.lstrip('/')
        self.high=self.number('memory.high');self.maximum=self.number('memory.max')
        finite=[v for v in (self.high,self.maximum) if v is not None]
        if not finite:raise RuntimeError('No finite user memory limit; specify an explicit policy before running')
        self.threshold=min(finite)
        self.admission=.80*self.threshold
        self.stop_launch=.85*self.threshold
        self.emergency=.93*self.threshold
    def number(self,name):
        s=(self.path/name).read_text().strip()
        return None if s=='max' else int(s)
    def snapshot(self):
        return {'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'memory_bytes':self.number('memory.current'),
                'swap_bytes':self.number('memory.swap.current')}


def estimate(job,old):
    """Conservative reservation from max of prior seeds/objectives and workload."""
    method=job['algorithm'];name=job['design'];p=job['options'];b=configurations(method)['baseline']
    prior=max(r['resources']['peak_rss_kib']*1024 for r in old.values()
              if r['algorithm']==method and r['design']==name)
    length=p['steps']/10
    if method=='mcts':factor=length*p['search-iterations']/12
    elif method=='hybrid':
        work=lambda x:x['warmup-steps']*x['warmup-episodes']+x['warmup-top-k']*max(x['steps']-x['warmup-steps'],0)*x['search-iterations']*7
        factor=work(p)/work(b)
    elif method=='sa':factor=length*(p['search-iterations']+(0 if 'initial-temperature' in p else 32))/(96+32)
    elif method in ('ucb1','ucb1-prefix'):factor=length*p['episodes']/24
    else:factor=length*p['episodes']/8*p['long-term-horizon']/3*p['long-term-rollouts']*max(1,p['early-stop-same-arm-streak']/3)
    # A config can alter which prefixes repeat. Reserve 25% beyond this estimate.
    return max(.45*GIB,prior*max(.6,factor)*1.25+.12*GIB)


class Race:
    def __init__(self,args):
        self.args=args;self.root=args.root.resolve();self.out=args.out.resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.previous=args.previous.resolve();self.abc=self.root/'abc';self.mem=Memory();self.cpus=physical_cpus(args.max_workers)
        original=json.loads((self.previous/'all_results.json').read_text())
        assert len(original)==216 and all(r['status']=='ok' for r in original)
        self.old={(r['objective'],r['algorithm'],r['design'],r['seed']):r for r in original}
        self.old_manifest=json.loads((self.previous/'manifest.json').read_text())
        if sha(self.abc)!=self.old_manifest['abc_sha256']:raise RuntimeError('ABC binary differs from control')
        self.rows={};self.new_done=0;self.started=time.time()
        self.manifest=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            abc_sha256=sha(self.abc),source_sha256=sha(self.root/'src/extSynSearch/synsearch.cpp'),
            driver_sha256=sha(Path(__file__)),previous=str(self.previous),actions=ACTIONS,
            configurations={a:configurations(a) for a in METHODS},train_designs=list(TRAIN),holdout_designs=list(HOLDOUT),
            tune_seeds=[0],validation_seeds=[1,2],mcts_seeds=[0],cpus=self.cpus,max_workers=len(self.cpus),
            cgroup_path=str(self.mem.path),memory_high_bytes=self.mem.high,memory_max_bytes=self.mem.maximum,
            reserved_budget_bytes=self.mem.admission,emergency_bytes=self.mem.emergency,
            search_timeout_sec=args.timeout,cec_timeout_sec=300,
            selection='Cartesian policy grid on or1200 -> top three plus baseline control on fir -> one per method/objective; mean target/baseline-seed0, time tie-break',
            memory_policy='Historical peak-based admission; max 8 independent ABC processes; kill only own largest active process if user usage reaches emergency threshold; retry once alone',
            comparability='All sequence lengths 10; iterations, episodes, warmup, rollouts, horizon and early-stop settings fixed per method. Actual evaluations and time can differ, especially SA auto-temperature calibration. Circuits were seen in earlier rounds: held out from THIS selection only.')
        path=self.out/'manifest.json'
        if path.exists():
            prior=json.loads(path.read_text())
            for k in ('abc_sha256','source_sha256','driver_sha256','configurations','actions'):
                assert prior[k]==self.manifest[k],k
            from datetime import datetime, timezone
            self.started=datetime.strptime(prior['created_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
        else:save(path,self.manifest)
        self.monitor=(self.out/'memory_monitor.jsonl').open('a',buffering=1)

    def job(self,phase,obj,method,cfg,design,seed):
        return dict(phase=phase,objective=obj,algorithm=method,config_id=cfg,design=design,seed=seed,
                    options=configurations(method)[cfg])

    def folder(self,job):
        return self.out/'runs'/job['objective']/job['algorithm']/job['config_id']/job['design']/f"seed-{job['seed']}"

    def known(self,job):
        k=key(job)
        if k in self.rows:return self.rows[k]
        if job['config_id']=='baseline':
            r=dict(job,status='ok',origin='previous',data=self.old[(job['objective'],job['algorithm'],job['design'],job['seed'])])
            self.rows[k]=r;return r
        path=self.folder(job)/'run.json'
        if path.exists():
            r=json.loads(path.read_text())
            if r['status']!='ok':
                save(path.parent/f"failed_run_attempt-{r.get('attempt',0)}.json",r)
                return None
            self.rows[k]=r;return r
        return None

    def launch(self,job,cpu,attempt):
        folder=self.folder(job);folder.mkdir(parents=True,exist_ok=True)
        existing=[int(p.name.split('-')[-1]) for p in folder.glob('attempt-*') if p.is_dir()]
        attempt=max([attempt]+[n+1 for n in existing])
        work=folder/f'attempt-{attempt}';work.mkdir(exist_ok=False)
        design=self.root/'benchmarks/VTR'/f"{job['design']}.abc.blif"
        native=work/'native.json';aig=work/'optimized.aig'
        options={'algorithm':job['algorithm'],'objective':job['objective'],**job['options'],
                 'seed':job['seed'],'actions':ACTIONS,'output':native}
        # Automatic temperature is the default; the CLI rejects explicit -1.
        if options.get('initial-temperature')==-1:
            del options['initial-temperature']
        command=f'read_blif {design}; strash; synsearch '+ ' '.join(f'--{k} {v}' for k,v in options.items())+f'; print_stats; write_aiger {aig}'
        (work/'command.txt').write_text(command+'\n')
        stream=(work/'search.log').open('w')
        process=subprocess.Popen(['/usr/bin/time','-f','%e %M','-o',str(work/'resources.txt'),
                 'taskset','-c',str(cpu),str(self.abc),'-s','-c',command],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        return dict(job=job,cpu=cpu,attempt=attempt,work=work,process=process,stream=stream,
                    started=time.monotonic(),reservation=min(estimate(job,self.old),self.mem.admission),reason=None)

    def end(self,task):
        proc=task['process'];proc.wait();task['stream'].close();work=task['work'];job=task['job']
        meta=dict(returncode=proc.returncode,reason=task['reason'],wall_sec=time.monotonic()-task['started'])
        save(work/'attempt.json',meta)
        row=dict(job,origin='new',status='failed',cpu=task['cpu'],attempt=task['attempt'],search=meta,
                 resources=resources(work/'resources.txt'),work=str(work))
        native=work/'native.json';aig=work/'optimized.aig'
        if proc.returncode==0 and native.exists() and aig.exists():
            try:
                result=json.loads(native.read_text());actual=stats(work/'search.log')
                base=json.loads((self.previous/'baselines'/job['design']/'baseline.json').read_text())['initial']
                assert (result['initial_and'],result['initial_lev'])==(base['nodes'],base['level'])
                assert (result['final_and'],result['final_lev'])==(actual['nodes'],actual['level'])
                assert actual['latches']==base['latches'] and result['objective']==job['objective']
                assert result['config']['sequence_length']==job['options']['steps']
                check=cec(self.abc,self.previous/'baselines'/job['design']/'input.aig',aig,work,task['cpu'],300)
                row['data']=dict(job,result=result,resources=row['resources'],cec=check,search=meta)
                row['status']='ok' if check['equivalent'] else 'cec_failed'
            except Exception as exc:row['error']=repr(exc)
        return row

    def progress(self,phase,total,done):
        save(self.out/'progress.json',dict(phase=phase,phase_total=total,phase_done=done,
             new_runs_finalized=self.new_done,elapsed_sec=time.time()-self.started,
             known_results=len(self.rows),memory=self.mem.snapshot()))

    def execute(self,phase,jobs):
        pending=[];done=0
        for job in jobs:
            if self.known(job) is not None:done+=1
            else:pending.append((job,0,False))
        # Big jobs first avoids starvation; fit smaller jobs in unused reservations.
        pending.sort(key=lambda item:estimate(item[0],self.old),reverse=True)
        active=[];total=len(jobs)
        print('PHASE',phase,'total',total,'cached',done,flush=True);self.progress(phase,total,done)
        while pending or active:
            snapshot=self.mem.snapshot();snapshot.update(phase=phase,active=len(active),pending=len(pending))
            self.monitor.write(json.dumps(snapshot)+'\n')
            if snapshot['memory_bytes']>=self.mem.emergency and active:
                # Stop the largest own active process group, never another user's task.
                victim=max(active,key=lambda t:self.group_rss(t['process'].pid))
                if victim['reason'] is None:
                    victim['reason']='memory_guard'
                    try:os.killpg(victim['process'].pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    print('MEMORY_GUARD',key(victim['job']),snapshot['memory_bytes'],flush=True)
            for task in list(active):
                if time.monotonic()-task['started']>self.args.timeout and task['process'].poll() is None:
                    task['reason']='timeout';os.killpg(task['process'].pid,signal.SIGKILL)
                if task['process'].poll() is not None:
                    active.remove(task);row=self.end(task)
                    if row['status']!='ok' and task['reason']=='memory_guard' and task['attempt']==0:
                        pending.insert(0,(task['job'],1,True));continue
                    self.rows[key(task['job'])]=row;save(self.folder(task['job'])/'run.json',row)
                    self.new_done+=1;done+=1;r=row.get('data',{}).get('result',{})
                    print('DONE',phase,f'{done}/{total}',*key(task['job']),row['status'],r.get('final_and'),r.get('final_lev'),row['resources'].get('process_wall_sec'),flush=True)
                    self.progress(phase,total,done)
            used={t['cpu'] for t in active};reserved=sum(t['reservation'] for t in active)
            snapshot=self.mem.snapshot()
            if snapshot['memory_bytes']<self.mem.stop_launch:
                for item in list(pending):
                    if len(active)>=len(self.cpus):break
                    job,attempt,alone=item;reserve=min(estimate(job,self.old),self.mem.admission)
                    if alone and active:continue
                    if any(t.get('exclusive') for t in active):break
                    if reserved+reserve>self.mem.admission:continue
                    # Also respect usage by unrelated processes in the same user slice.
                    rss=sum(self.group_rss(t['process'].pid) for t in active)
                    background=max(snapshot['memory_bytes']-rss,0)
                    if background+reserved+reserve>self.mem.stop_launch:continue
                    cpu=next(c for c in self.cpus if c not in used)
                    task=self.launch(job,cpu,attempt);task['exclusive']=alone
                    active.append(task);pending.remove(item);used.add(cpu);reserved+=reserve
                    print('START',phase,*key(job),'reserve_GiB',round(reserve/GIB,2),'active',len(active),flush=True)
                    if alone:break
            time.sleep(1)
        self.progress(phase,total,done)

    @staticmethod
    def group_rss(pgid):
        total=0
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                s=path.read_text();parts=s[s.rfind(')')+2:].split()
                if int(parts[2])==pgid:total+=int(parts[21])*os.sysconf('SC_PAGE_SIZE')
            except (OSError,ValueError,IndexError):continue
        return total

    def rank(self,obj,method,configs,designs):
        ranked=[]
        for cfg in configs:
            ratios=[];times=[];valid=True
            for design in designs:
                r=self.known(self.job('selection',obj,method,cfg,design,0))
                if not r or r['status']!='ok':valid=False;break
                field='final_and' if obj=='nodes' else 'final_lev'
                ratios.append(r['data']['result'][field]/self.old[(obj,method,design,0)]['result'][field])
                times.append(r['data']['resources']['process_wall_sec'])
            if valid:ranked.append(dict(config_id=cfg,mean_ratio=mean(ratios),mean_process_sec=mean(times)))
        return sorted(ranked,key=lambda r:(round(r['mean_ratio'],12),r['mean_process_sec'],r['config_id']))

    def main(self):
        pairs=[(o,a) for o in ('nodes','level') for a in METHODS]
        self.execute('screen',[self.job('screen',o,a,c,TRAIN[0],0) for o,a in pairs for c in configurations(a)])
        promoted={f'{o}/{a}':self.rank(o,a,configurations(a),TRAIN[:1])[:3] for o,a in pairs}
        save(self.out/'promoted.json',promoted)
        self.execute('confirm',[self.job('confirm',o,a,r['config_id'],TRAIN[1],0)
             for o,a in pairs for r in promoted[f'{o}/{a}']])
        selection={}
        for o,a in pairs:
            ranked=self.rank(o,a,list(dict.fromkeys([r['config_id'] for r in promoted[f'{o}/{a}']]+['baseline'])),TRAIN)
            if not ranked:raise RuntimeError('No valid finalist')
            selection[f'{o}/{a}']={'selected':ranked[0]['config_id'],'ranking':ranked,
                'options':configurations(a)[ranked[0]['config_id']]}
        # Frozen before reading any newly evaluated holdout result or seed 1/2.
        save(self.out/'selection.json',selection)
        print('FROZEN_SELECTION',json.dumps(selection),flush=True)
        jobs=[self.job('validation',o,a,selection[f'{o}/{a}']['selected'],d,s)
              for o,a in pairs for d in DESIGNS for s in ([0] if a=='mcts' else [0,1,2])]
        self.execute('validation',jobs)
        final=[self.known(j) for j in jobs]
        save(self.out/'final_results.json',final)
        save(self.out/'all_trials.json',list(self.rows.values()))
        save(self.out/'completed.json',dict(final_cases=len(final),new_runs=sum(r['origin']=='new' for r in self.rows.values()),
             finalized_this_process=self.new_done,
             final_failures=sum(r['status']!='ok' for r in final),
             trial_failures=sum(r['status']!='ok' for r in self.rows.values()),elapsed_sec=time.time()-self.started))
        self.monitor.close()
        print('FINISHED',len(final),'final cases',sum(r['status']!='ok' for r in final),'failures',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-workers',type=int,default=8);p.add_argument('--timeout',type=int,default=900)
    args=p.parse_args()
    if args.max_workers<1:raise SystemExit('max-workers must be positive')
    Race(args).main()

if __name__=='__main__':main()
