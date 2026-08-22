"""Confirm the selected 128-wide Shakespeare transformer across three seeds."""
from __future__ import annotations
import json, math, statistics, time
from datetime import UTC, datetime
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from shakespeare_transformer import DEFAULT_DATA, TinyTransformerLanguageModel, atomic_json_write, estimate_loss, generate_from_prompt, get_batch, load_data, loss_fn
from paths import LOCAL_EXPERIMENTS_DIR, available_run, local_result
ROOT=Path(__file__).resolve().parents[1]; OUT=local_result('shakespeare-final.json'); EXP=LOCAL_EXPERIMENTS_DIR; STEPS=3000; CONTEXT=64; WIDTH=128; BLOCKS=2; HEADS=4

def eval_loss(model,data,seed):
 vals=[estimate_loss(model,data,seed=200042+i,batches=20,batch_size=32,context_size=CONTEXT) for i in range(5)]
 return statistics.mean(vals),statistics.pstdev(vals)

def train(seed,train,val,vocab,c2i):
 run_id=f'shakespeare-final-seed-{seed:03d}'; run_dir=EXP/run_id
 if seed==42: run_dir=available_run('shakespeare-scale-width-001'); model=TinyTransformerLanguageModel(len(vocab),CONTEXT,WIDTH,HEADS,BLOCKS); model.load_weights(str(run_dir/'checkpoint-3000.safetensors')); mx.eval(model.parameters()); elapsed=json.loads((run_dir/'config.json').read_text())['elapsedSeconds']; peak=json.loads((run_dir/'config.json').read_text())['peakMetalMemoryBytes']; reused=True
 else:
  mx.reset_peak_memory(); mx.random.seed(seed); model=TinyTransformerLanguageModel(len(vocab),CONTEXT,WIDTH,HEADS,BLOCKS); mx.eval(model.parameters()); schedule=optim.join_schedules([optim.linear_schedule(.0001,.001,100),optim.cosine_decay(.001,STEPS-100,end=.0001)],[100]); opt=optim.AdamW(learning_rate=schedule,weight_decay=.01); vg=nn.value_and_grad(model,loss_fn); run_dir.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
  for step in range(1,STEPS+1):
   mx.random.seed(seed+step); x,y=get_batch(train,32,CONTEXT); loss,grads=vg(model,x,y); opt.update(model,grads); mx.eval(model.parameters(),opt.state,loss)
  elapsed=time.perf_counter()-started; peak=mx.get_peak_memory(); model.save_weights(str(run_dir/'checkpoint-3000.safetensors')); reused=False
 vl,std=eval_loss(model,val,seed); continuation=generate_from_prompt(model,'To be, or not to be',c2i,vocab,seed=300042,characters=160,temperature=.8); params=sum(p.size for _,p in nn.utils.tree_flatten(model.parameters()))
 result={'runId':run_dir.name,'seed':seed,'reusedScalingRun':reused,'parameterCount':params,'elapsedSeconds':elapsed,'peakMetalMemoryBytes':peak,'validationLossMean':vl,'validationLossStd':std,'perplexity':math.exp(vl),'continuation':continuation}
 if not reused: (run_dir/'config.json').write_text(json.dumps({'runId':run_dir.name,'seed':seed,'steps':STEPS,'batchSize':32,'contextSize':CONTEXT,'modelSize':WIDTH,'attentionHeads':HEADS,'transformerBlocks':BLOCKS,'learningRateRecipe':'100-step warmup then cosine 0.001 to 0.0001','weightDecay':.01,'parameterCount':params,'elapsedSeconds':elapsed,'peakMetalMemoryBytes':peak,'createdAt':datetime.now(UTC).isoformat()},indent=2)+'\n')
 print(seed,vl,std,flush=True); return result

def main():
 train_data,val,vocab,c2i=load_data(DEFAULT_DATA); payload={'status':'Running','experimentId':'shakespeare-final-001','updatedAt':datetime.now(UTC).isoformat(),'architecture':{'modelSize':WIDTH,'transformerBlocks':BLOCKS,'attentionHeads':HEADS,'contextSize':CONTEXT,'parameterCount':420673},'seeds':[]}; atomic_json_write(OUT,payload); started=time.perf_counter()
 for seed in (42,43,44): payload['seeds'].append(train(seed,train_data,val,vocab,c2i)); payload['updatedAt']=datetime.now(UTC).isoformat(); atomic_json_write(OUT,payload)
 losses=[x['validationLossMean'] for x in payload['seeds']]; payload['validationLossAcrossSeedsMean']=statistics.mean(losses); payload['validationLossAcrossSeedsStd']=statistics.pstdev(losses); payload['perplexityFromMeanLoss']=math.exp(payload['validationLossAcrossSeedsMean']); payload['selectedRunId']=min(payload['seeds'],key=lambda x:x['validationLossMean'])['runId']; payload['elapsedSeconds']=time.perf_counter()-started; payload['status']='Complete'; payload['updatedAt']=datetime.now(UTC).isoformat(); atomic_json_write(OUT,payload); d=EXP/'shakespeare-final-001'; d.mkdir(exist_ok=True); (d/'config.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='seeds'},indent=2)+'\n')
if __name__=='__main__': main()
