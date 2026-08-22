"""Compare width, depth, and context scaling with one fixed training recipe."""
from __future__ import annotations
import json, math, statistics, time
from datetime import UTC, datetime
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from shakespeare_transformer import DEFAULT_DATA, TinyTransformerLanguageModel, atomic_json_write, estimate_loss, generate_from_prompt, get_batch, load_data, loss_fn
from paths import LOCAL_EXPERIMENTS_DIR, local_result

ROOT=Path(__file__).resolve().parents[1]; OUT=local_result('shakespeare-scaling.json'); EXP=LOCAL_EXPERIMENTS_DIR; SEED=42; STEPS=3000
VARIANTS=[
 ('shakespeare-scale-reference-001','Reference','Reference 64×2 · context 64',64,2,64,4),
 ('shakespeare-scale-width-001','Width','Wider 128×2 · context 64',128,2,64,4),
 ('shakespeare-scale-depth-001','Depth','Deeper 64×4 · context 64',64,4,64,4),
 ('shakespeare-scale-context-001','Context','Longer context 64×2 · context 128',64,2,128,4),
]

def evaluate(model,data,seed,context):
 vals=[estimate_loss(model,data,seed=seed+i,batches=20,batch_size=32,context_size=context) for i in range(5)]
 return statistics.mean(vals),statistics.pstdev(vals)

def run(spec,train,val,vocab,c2i):
 run_id,factor,label,width,blocks,context,heads=spec; mx.reset_peak_memory(); mx.random.seed(SEED)
 model=TinyTransformerLanguageModel(len(vocab),context,width,heads,blocks); mx.eval(model.parameters())
 params=sum(p.size for _,p in nn.utils.tree_flatten(model.parameters()))
 schedule=optim.join_schedules([optim.linear_schedule(.0001,.001,100),optim.cosine_decay(.001,STEPS-100,end=.0001)],[100])
 opt=optim.AdamW(learning_rate=schedule,weight_decay=.01); value_grad=nn.value_and_grad(model,loss_fn); checkpoints=[]; started=time.perf_counter(); run_dir=EXP/run_id; run_dir.mkdir(parents=True,exist_ok=True)
 def capture(step,loss):
  vl=estimate_loss(model,val,seed=SEED+500000+step,batches=30,batch_size=32,context_size=context); now=datetime.now(UTC).isoformat()
  checkpoints.append({'step':step,'capturedAt':now,'trainLoss':loss,'validationLoss':vl,'learningRate':float(opt.learning_rate.item()) if step else .0001}); model.save_weights(str(run_dir/f'checkpoint-{step:04d}.safetensors')); print(label,step,vl,flush=True)
 capture(0,None)
 for step in range(1,STEPS+1):
  mx.random.seed(SEED+step); x,y=get_batch(train,32,context); loss,grads=value_grad(model,x,y); opt.update(model,grads); mx.eval(model.parameters(),opt.state,loss)
  if step in (250,1000,2000,3000): capture(step,float(loss.item()))
 elapsed=time.perf_counter()-started; tr,_=evaluate(model,train,SEED+100000,context); vl,std=evaluate(model,val,SEED+200000,context)
 continuation=generate_from_prompt(model,'To be, or not to be',c2i,vocab,seed=SEED+300000,characters=120,temperature=.8)
 result={'runId':run_id,'factor':factor,'label':label,'modelSize':width,'transformerBlocks':blocks,'contextSize':context,'attentionHeads':heads,'parameterCount':params,'elapsedSeconds':elapsed,'peakMetalMemoryBytes':mx.get_peak_memory(),'checkpoints':checkpoints,'evaluation':{'trainLossMean':tr,'validationLossMean':vl,'validationLossStd':std,'generalisationGap':vl-tr,'perplexity':math.exp(vl),'continuation':continuation}}
 (run_dir/'config.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('checkpoints','evaluation')},indent=2)+'\n')
 return result

def main():
 train,val,vocab,c2i=load_data(DEFAULT_DATA); payload={'status':'Running','experimentId':'shakespeare-scaling-001','updatedAt':datetime.now(UTC).isoformat(),'controlledVariables':{'seed':SEED,'steps':STEPS,'batchSize':32,'learningRateRecipe':'100-step warmup then cosine 0.001 to 0.0001','weightDecay':.01,'evaluationRepeats':5,'evaluationBatchesPerRepeat':20},'variants':[]}; atomic_json_write(OUT,payload); started=time.perf_counter()
 for spec in VARIANTS:
  payload['variants'].append(run(spec,train,val,vocab,c2i)); payload['updatedAt']=datetime.now(UTC).isoformat(); atomic_json_write(OUT,payload)
 payload['status']='Complete'; payload['elapsedSeconds']=time.perf_counter()-started; payload['selectedRunId']=min(payload['variants'],key=lambda x:x['evaluation']['validationLossMean'])['runId']; payload['updatedAt']=datetime.now(UTC).isoformat(); atomic_json_write(OUT,payload)
 d=EXP/'shakespeare-scaling-001'; d.mkdir(exist_ok=True); (d/'config.json').write_text(json.dumps({'experimentId':payload['experimentId'],'selectedRunId':payload['selectedRunId'],'elapsedSeconds':payload['elapsedSeconds'],'controlledVariables':payload['controlledVariables']},indent=2)+'\n')
if __name__=='__main__': main()
