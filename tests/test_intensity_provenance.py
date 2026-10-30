"""Reject stale references and invalid selections in intensity-based comparisons."""
import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from coarse_tpp import easytpp as E, intensity_eval as I, intensity_exceedance as X, rescore_retrained as R


def reference_case(tmp_path):
    inp = {k + '_sha256': k + '-current' for k in ('checkpoint', 'config', 'data', 'result')}
    cell = dict(dataset='retweet_sub', model='NHP', seed=0, experiment='joint', inputs=inp,
                dev={'floor_1e-6': -1., 'joint_floor_1e-6': -2.},
                test={'floor_1e-6': -1., 'joint_floor_1e-6': -2.})
    ident = dict(version=1, experiment='joint', dataset='retweet_sub', model='NHP', seed=0,
                 ours_key='joint_floor_1e-6', repeats=50, inputs=inp)
    rec = dict(identity=ident, checkpoint_sha256=inp['checkpoint_sha256'], experiment='joint',
               ours_key='joint_floor_1e-6', repeats=50,
               dev=dict(mc_mean=-2., mc_se_mean=0.), test=dict(mc_mean=-2., mc_se_mean=0.))
    root = tmp_path/'repro'; root.mkdir()
    p = root/'retweet_sub__NHP__s0.json'
    return cell, rec, p


@pytest.mark.parametrize('field', ['config', 'data', 'result'])
def test_repro_gate_rejects_changed_inputs(tmp_path, field):
    cell, rec, p = reference_case(tmp_path)
    rec['identity']['inputs'] = dict(rec['identity']['inputs'], **{field+'_sha256': 'old'})
    p.write_text(json.dumps(rec))
    assert R.mc_gate(cell, tmp_path, [])
    assert not cell['repro']['ok']


def test_repro_gate_rejects_wrong_objective_even_if_time_matches(tmp_path):
    cell, rec, p = reference_case(tmp_path)
    rec.update(experiment='time_only', ours_key='floor_1e-6')
    rec['identity'].update(experiment='time_only', ours_key='floor_1e-6')
    rec['dev']['mc_mean'] = rec['test']['mc_mean'] = -1.
    p.write_text(json.dumps(rec))
    assert R.mc_gate(cell, tmp_path, [])
    assert not cell['repro']['ok']


def test_repro_gate_requires_provenance_and_accepts_matching_reference(tmp_path):
    cell, rec, p = reference_case(tmp_path)
    p.write_text(json.dumps(rec))
    assert R.mc_gate(cell, tmp_path, []) == [] and cell['repro']['ok']
    rec.pop('identity')
    p.write_text(json.dumps(rec))
    assert R.mc_gate(cell, tmp_path, []) and not cell['repro']['ok']


def test_selection_rejects_quarantine_but_keeps_diverged_runs():
    cells = [dict(dataset='retweet_sub', model=m, seed=s, dev={R.F6:v,R.IV:v}, test={R.F6:-101.},
                  quarantined=bad) for m,v,bad in [('NHP',-2.,False),('S2P2',5.,True)] for s in E.SEEDS]
    sel, _ = X.select(cells, 'retweet_sub')
    assert sel['floor'][0] == sel['interval'][0] == 'NHP'
    for c in cells:
        if c['model']=='NHP': c['seed'] += 5
    sel, _ = X.select(cells, 'retweet_sub')
    assert sel['floor'][0] is None


def forecast_case(tmp_path):
    out = tmp_path/'forecast'; out.mkdir()
    root = tmp_path/'runs'; loc = root/'retweet_sub'/'NHP'/'seed0'; loc.mkdir(parents=True)
    cfg=dict(model_config=dict(hidden_size=8,time_emb_size=4,num_layers=1,num_heads=1,
                              num_event_types=2,num_event_types_pad=3,event_pad_index=2,model_id='NHP'))
    model=I.build_model('NHP',cfg,torch.device('cpu'))
    torch.save(model.state_dict(),loc/'saved_model.pt')
    (loc/'config.yaml').write_text(yaml.safe_dump(cfg))
    (loc/'result.json').write_text('{}')
    data = tmp_path/'data.pkl'
    events=[[dict(time_since_start=t,time_since_last_event=g,type_event=k)
             for t,g,k in [(0.,0.,0),(.2,.2,1),(.8,.6,0)]]]
    data.write_bytes(pickle.dumps(dict(train=events,dev=events,test=events,dim_process=2)))
    arr = out/'retweet_sub__NHP__s0.npz'
    rec=X.score_run('retweet_sub','NHP',0,1.,[0,1,2,3],I.Rule(),root,data,tmp_path/'scores',out,torch.device('cpu'),check=False)
    return out,root,data,arr,rec


@pytest.mark.parametrize('changed', ['data','arrays'])
def test_forecast_cache_rejects_changed_data_or_arrays(tmp_path, changed):
    out,root,data,arr,rec = forecast_case(tmp_path)
    assert X.cache_ok(rec,'retweet_sub','NHP',0,I.Rule(),[0,1,2,3],1.,root)
    (data if changed=='data' else arr).write_bytes(b'changed')
    assert not X.cache_ok(rec,'retweet_sub','NHP',0,I.Rule(),[0,1,2,3],1.,root)


def test_report_requires_current_selection_inputs(tmp_path):
    out,root,data,arr,rec = forecast_case(tmp_path)
    (out/'retweet_sub__NHP__s0.json').write_text(json.dumps(rec))
    (out/'retweet_sub__selection.json').write_text(json.dumps(dict(
        selection={'floor':['NHP',-1.],'interval':['NHP',-1.]},delta=1.,
        h_grid={k:dict(kh=i,h=float(i)) for i,k in enumerate(X.A.H_KEYS)})))
    a = SimpleNamespace(out=out, ckpt=root, scores=tmp_path/'missing',strata=65536,strat_seed=0)
    with pytest.raises(ValueError,match='selection|score'):
        X.cmd_report(a)


def scored_forecast(tmp_path):
    out,root,data,arr,rec=forecast_case(tmp_path)
    prep=tmp_path/'prepared'; prep.mkdir()
    (prep/'data_retweet_sampled.pkl').write_bytes(data.read_bytes())
    scores=tmp_path/'scores'; (scores/'scores').mkdir(parents=True)
    for seed in E.SEEDS:
        cell=dict(experiment='time_only',dataset='retweet_sub',model='NHP',seed=seed,
                  cache_key=dict(version=R.VERSION),inputs={},problems=[],
                  dev={R.F6:-1.,R.IV:-2.},test={R.F6:-1.})
        (scores/'scores'/f'retweet_sub__NHP__s{seed}.json').write_text(json.dumps(cell))
    audit=tmp_path/'audit.json'; audit.write_text(json.dumps([dict(dataset='retweet_sub',delta=1.)]))
    a=SimpleNamespace(out=out,ckpt=root,data=prep,scores=scores,audit=audit,datasets=['retweet_sub'],
                      models=['NHP'],seeds=[0],strata=65536,strat_seed=0,cpu=True,
                      allow_fullynn=False,force=False,max_seqs=None,no_check=True)
    X.cmd_score(a)
    return a


def test_report_marks_missing_seeds_incomplete(tmp_path):
    a=scored_forecast(tmp_path)
    X.cmd_report(a)
    rows=json.loads((a.out/'summary.json').read_text())
    assert rows[0]['seeds']==[0] and rows[0]['complete'] is False


def test_report_rejects_changed_selection_scores(tmp_path):
    a=scored_forecast(tmp_path)
    p=a.scores/'scores'/'retweet_sub__NHP__s0.json'
    c=json.loads(p.read_text()); c['dev'][R.F6]=8.; p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='selection|score'):
        X.cmd_report(a)
