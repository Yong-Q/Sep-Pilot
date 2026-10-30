"""Controlled public CSV packet and plot script; no legacy test collection."""
import pytest
from agents import report_authoring

SCRIPT='''import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with open('/inputs/table0.csv') as f:
    rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(4,3))
ax.bar([x['material'] for x in rows],[float(x['loading_mol_kg']) for x in rows])
ax.set_ylabel('Loading (mol/kg)')
fig.savefig('/output/loading.png')
'''

@pytest.fixture
def packet(tmp_path):
    root=tmp_path/'runs'/'owner'/'conversation';root.mkdir(parents=True)
    table=root/'actual.csv';table.write_text('material,loading_mol_kg,method\nA,0.75,GCMC\nB,1.25,GCMC\n')
    _,fact=report_authoring._read(root,table,{'bytes':0})
    p={'schema_version':1,'owner':{'username':'owner','conv_id':'conversation'},'root':str(root),
       'report_step':'figures','report_token':'token','instructions':'Show the actual measured loadings.',
       'sources':[{'id':'S1','step_id':'measure','token':'fixture-native-token','call_id':'fixture-call','source_sha256':fact['sha256']}],
       'file_facts':[fact],'tables':[{'id':'T1','source_id':'S1','path':'actual.csv','sha256':fact['sha256'],
        'columns':['material','loading_mol_kg','method'],'rows':[{'material':'A','loading_mol_kg':'0.75','method':'GCMC'},
         {'material':'B','loading_mol_kg':'1.25','method':'GCMC'}]}]}
    p['packet_sha256']=report_authoring._digest(p)
    return p
