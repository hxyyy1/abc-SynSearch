#!/usr/bin/env python3
"""Reproducible VTR single-objective experiment; one ABC process per search.

Six methods x six VTR circuits x two objectives x three seeds. Writes raw
networks, commands, logs, resource metrics and separate combinational CEC logs.
No claim of equal wall-clock/evaluation budget is made across methods.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import signal
import subprocess
import time

METHODS = ('mcts', 'hybrid', 'sa', 'ucb1', 'ucb1-prefix', 'linucb')
ACTIONS = 'balance,rewrite,rewrite-z,refactor,refactor-z,resub,resub-z'
RESYN2 = 'balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance'
PARAMETERS = dict(sequence_length=10, mcts_iterations=12, sa_iterations=96,
                  episodes=24, linucb_episodes=8, warmup_steps=3, warmup_episodes=16, warmup_top_k=1,
                  rollouts=1, horizon=3, streak=3)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def run(command, log, timeout):
    started = time.perf_counter()
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            rc = process.wait(timeout=timeout)
            return dict(returncode=rc, timeout=False, wall_sec=time.perf_counter()-started)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return dict(returncode=process.returncode, timeout=True, wall_sec=time.perf_counter()-started)


def stats(log):
    text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text())
    hits = re.findall(r'lat\s*=\s*(\d+).*?and\s*=\s*(\d+).*?lev\s*=\s*(\d+)', text)
    if not hits:
        raise RuntimeError(f'No AIG statistics: {log}')
    lat, nodes, lev = map(int, hits[-1])
    return dict(nodes=nodes, level=lev, latches=lat)


def resources(path):
    if not path.exists():
        return {}
    try:
        wall, rss = path.read_text().splitlines()[-1].split()
        return dict(process_wall_sec=float(wall), peak_rss_kib=int(rss))
    except (ValueError, IndexError):
        return {}


def cec(abc, original, final, folder, cpu, timeout):
    script = f'cec -n {original} {final}'
    (folder/'cec_command.txt').write_text(script+'\n')
    meta = run(['taskset','-c',str(cpu),str(abc),'-s','-c',script], folder/'cec.log', timeout)
    text = (folder/'cec.log').read_text()
    meta['equivalent'] = meta['returncode']==0 and 'Networks are equivalent' in text
    return meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--timeout',type=int,default=1800)
    ap.add_argument('--cec-timeout',type=int,default=300)
    args=ap.parse_args()
    root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    abc=root/'abc';designs=sorted((root/'benchmarks/VTR').glob('*.blif'))
    assert len(designs)==6
    allowed=os.sched_getaffinity(0)
    topology=subprocess.check_output(['lscpu','-p=CPU,CORE'],text=True)
    seen=set();cpus=[]
    for line in topology.splitlines():
        if line.startswith('#'):continue
        cpu,core=map(int,line.split(','))
        if cpu in allowed and core not in seen:
            seen.add(core);cpus.append(cpu)
    cpus=cpus[:args.workers]
    cpu_queue=queue.Queue()
    for cpu in cpus:cpu_queue.put(cpu)
    manifest=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  hostname=platform.node(),platform=platform.platform(),
                  lscpu=subprocess.check_output(['lscpu'],text=True),
                  compiler=subprocess.check_output(['g++','--version'],text=True),
                  build='make -j16 ABC_USE_NO_READLINE=1 OPTFLAGS=-O2',
                  abc_sha256=sha(abc),source_sha256=sha(root/'src/extSynSearch/synsearch.cpp'),
                  driver_sha256=sha(Path(__file__)),parameters=PARAMETERS,actions=ACTIONS,
                  objectives=['nodes','level'],seeds=list(map(int,args.seeds.split(','))),
                  user_memory_high_gib=12,user_memory_max_gib=16,pilot_note='Earlier six-worker higher-budget pilot aborted after swap pressure; pilot excluded',
                  cpus=cpus,workers=len(cpus),timeout_sec=args.timeout,cec_timeout_sec=args.cec_timeout,
                  designs=[dict(name=p.name,sha256=sha(p),size=p.stat().st_size) for p in designs],
                  equivalence_scope='Combinational equivalence with matching latch boundaries; no retiming performed',
                  objective_note='Explicit objectives; LinUCB ties use other metric; node spike guard remains 1.8')
    mp=out/'manifest.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        for k in ('abc_sha256','source_sha256','driver_sha256','parameters','actions','seeds','designs'):
            if old[k]!=manifest[k]:raise RuntimeError(f'Resume manifest mismatch: {k}')
    else:save(mp,manifest)
    baselines={}
    for design in designs:
        name=design.name.removesuffix('.abc.blif');folder=out/'baselines'/name
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'baseline.json').exists():
            baselines[name]=json.loads((folder/'baseline.json').read_text());continue
        raw=folder/'input.aig';resyn=folder/'resyn2.aig'
        log=folder/'input.log'
        script=f'read_blif {design}; strash; print_stats; write_aiger {raw}'
        r=run(['taskset','-c',str(cpus[0]),str(abc),'-s','-c',script],log,args.timeout)
        if r['returncode'] or not raw.exists():raise RuntimeError(str(log))
        initial=stats(log)
        script=f'read_blif {design}; strash; {RESYN2}; print_stats; write_aiger {resyn}'
        (folder/'resyn2_command.txt').write_text(script+'\n')
        r=run(['/usr/bin/time','-f','%e %M','-o',str(folder/'resources.txt'),
               'taskset','-c',str(cpus[0]),str(abc),'-s','-c',script],folder/'resyn2.log',args.timeout)
        if r['returncode'] or not resyn.exists():raise RuntimeError(str(folder))
        baseline=dict(design=name,initial=initial,resyn2=stats(folder/'resyn2.log'),
                      resources=resources(folder/'resources.txt'),
                      cec=cec(abc,raw,resyn,folder,cpus[0],args.cec_timeout))
        save(folder/'baseline.json',baseline);baselines[name]=baseline
        print('BASELINE',name,json.dumps(baseline),flush=True)
    jobs=[]
    for seed in manifest['seeds']:
        for objective in ('nodes','level'):
            for design in designs:
                for method in METHODS:jobs.append((seed,objective,design,method))
    def execute(job):
        seed,objective,design,method=job;name=design.name.removesuffix('.abc.blif')
        folder=out/'runs'/objective/method/name/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
        status=folder/'status.json'
        if status.exists():return json.loads(status.read_text())
        cpu=cpu_queue.get()
        try:
            native=folder/'native.json';aig=folder/'optimized.aig'
            opts={'algorithm':method,'objective':objective,'steps':PARAMETERS['sequence_length'],
                  'seed':seed,'actions':ACTIONS,
                  'search-iterations':PARAMETERS['sa_iterations'] if method=='sa' else PARAMETERS['mcts_iterations'],
                  'episodes':PARAMETERS['linucb_episodes'] if method=='linucb' else PARAMETERS['episodes'],'warmup-steps':PARAMETERS['warmup_steps'],
                  'warmup-episodes':PARAMETERS['warmup_episodes'],'warmup-top-k':PARAMETERS['warmup_top_k'],
                  'long-term-rollouts':PARAMETERS['rollouts'],'long-term-horizon':PARAMETERS['horizon'],
                  'early-stop-same-arm-streak':PARAMETERS['streak'],'output':native}
            flags=' '.join(f'--{k} {v}' for k,v in opts.items())
            script=f'read_blif {design}; strash; synsearch {flags}; print_stats; write_aiger {aig}'
            (folder/'command.txt').write_text(script+'\n')
            runmeta=run(['/usr/bin/time','-f','%e %M','-o',str(folder/'resources.txt'),
                         'taskset','-c',str(cpu),str(abc),'-s','-c',script],folder/'search.log',args.timeout)
            row=dict(design=name,objective=objective,algorithm=method,seed=seed,cpu=cpu,
                     search=runmeta,resources=resources(folder/'resources.txt'),status='failed')
            if runmeta['returncode']==0 and native.exists() and aig.exists():
                result=json.loads(native.read_text());row['result']=result
                actual=stats(folder/'search.log');base=baselines[name]['initial']
                assert actual['nodes']==result['final_and'] and actual['level']==result['final_lev']
                assert actual['latches']==base['latches']
                assert (result['initial_and'],result['initial_lev'])==(base['nodes'],base['level'])
                assert result['objective']==objective
                row['cec']=cec(abc,out/'baselines'/name/'input.aig',aig,folder,cpu,args.cec_timeout)
                row['status']='ok' if row['cec']['equivalent'] else 'cec_failed'
            save(status,row)
            return row
        except Exception as exc:
            row=dict(design=name,objective=objective,algorithm=method,seed=seed,status='error',error=repr(exc))
            save(status,row);return row
        finally:cpu_queue.put(cpu)
    done=[]
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures=[pool.submit(execute,job) for job in jobs]
        for future in as_completed(futures):
            row=future.result();done.append(row)
            r=row.get('result',{})
            print(f'{len(done)}/{len(jobs)}',row['objective'],row['algorithm'],row['design'],row['seed'],row['status'],
                  r.get('final_and'),r.get('final_lev'),row.get('resources',{}).get('process_wall_sec'),flush=True)
            save(out/'progress.json',dict(completed=len(done),total=len(jobs),
                  successful=sum(x['status']=='ok' for x in done),last=row))
    save(out/'all_results.json',done)
    fields=['objective','design','algorithm','seed','status','initial_and','initial_lev',
            'final_and','final_lev','total_runtime_sec','evaluations','cache_hits','process_wall_sec','peak_rss_kib','recipe']
    with (out/'all_runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in sorted(done,key=lambda r:(r['objective'],r['design'],r['algorithm'],r['seed'])):
            writer.writerow(dict(row,**row.get('result',{}),**row.get('resources',{})))
    failures=[x for x in done if x['status']!='ok']
    print('FINISHED',len(done),'failures',len(failures),flush=True)
    return bool(failures)

if __name__=='__main__':raise SystemExit(main())
