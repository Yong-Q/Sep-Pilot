import hashlib
from pathlib import Path
import pytest
from PIL import Image
from PyPDF2 import PdfReader
from agents.plot_sandbox import plot_result_code

def source(root):
    path=root/'plot.png'; Image.new('RGB',(40,30),'blue').save(path)
    return path

def launch(root, image, script, directory='attempt', sha=None):
    return plot_result_code(script,input_files=[{'path':str(image),'input_id':'figure','format':'png',
        'expected_sha256':sha or hashlib.sha256(image.read_bytes()).hexdigest()}],
        output_dir=str(root/directory),owned_root=root,_mode='data',
        _declared_outputs=[{'name':'report.pdf','format':'file'}],_owner={'username':'u','conv_id':'c'})

def test_owned_image_to_pdf_in_real_isolated_sandbox(tmp_path):
    root=tmp_path/'runs/u/c';root.mkdir(parents=True); image=source(root);before=image.read_bytes()
    script='''import matplotlib.pyplot as plt
from PIL import Image
fig,ax=plt.subplots()
ax.imshow(Image.open('/inputs/figure.png'))
ax.set_title('Task report');ax.set_xlabel('DOI: 10.1000/example')
fig.savefig('/output/report.pdf')
'''
    result=launch(root,image,script)
    assert result['ok'] is True and result['sandbox']['os_isolated'] is True
    artifact=result['artifacts'][0];pdf=Path(artifact['path'])
    assert pdf.name=='report.pdf' and pdf.is_relative_to(root/'attempt/artifacts')
    assert artifact['sha256']==hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert artifact['source_refs']==[hashlib.sha256(before).hexdigest()]
    reader=PdfReader(str(pdf),strict=True)
    assert len(reader.pages)==1 and 'Task report' in reader.pages[0].extract_text()
    resources=reader.pages[0]['/Resources'].get_object()
    images=resources['/XObject'].get_object().values()
    assert any(item.get_object().get('/Subtype')=='/Image' for item in images)
    assert image.read_bytes()==before
    with pytest.raises((ValueError,FileExistsError,OSError)):
        launch(root,image,script)

def test_wrong_source_hash_does_not_publish_and_file_format_is_not_a_gate(tmp_path):
    root=tmp_path/'runs/u/c';root.mkdir(parents=True);image=source(root)
    with pytest.raises(ValueError,match='SHA'):
        launch(root,image,"open('/output/report.pdf','wb').write(b'bad')",sha='0'*64)
    result=launch(root,image,"open('/output/report.pdf','wb').write(b'%PDF-fake')",directory='bad')
    assert result['ok'] is True and (root/'bad/artifacts/report.pdf').read_bytes()==b'%PDF-fake'

def test_arbitrary_formats_and_extensionless_in_same_sandbox(tmp_path):
    root=tmp_path/'runs/u/c';root.mkdir(parents=True); image=source(root)
    outputs=[{'name':name,'format':'file'} for name in ['report.docx','diagram.svg','noext']]
    script="open('/output/report.docx','wb').write(b'\\x00docx')\nopen('/output/diagram.svg','w').write('<svg/>')\nopen('/output/noext','w').write('ok')"
    result=plot_result_code(script,input_files=[{'path':str(image),'input_id':'figure','format':'png'}],
        output_dir=str(root/'any'),owned_root=root,_mode='data',_declared_outputs=outputs,_owner={'username':'u','conv_id':'c'})
    assert result['ok'] is True and {item['name'] for item in result['artifacts']}=={x['name'] for x in outputs}

from tests.test_parallel_workflow import factory,node,eventually

def test_real_dag_pdf_is_downloadable_only_by_its_owner(factory,tmp_path,monkeypatch):
    from agents import registry,workspace
    from agents.result_delivery import list_deliverables,read_deliverable
    root=tmp_path/'runs/tester/pdf-dag'; image=root/'source.png';out=root/'derived';pdf=out/'artifacts/report.pdf'
    script="import matplotlib.pyplot as p\nfrom PIL import Image\nf,a=p.subplots();a.imshow(Image.open('/inputs/figure.png'));a.set_title('PDF');f.savefig('/output/report.pdf')"
    producer=node('make_pdf',tool='run_analysis_sandbox',args={'script':script,
        'input_files':[{'path':str(image),'input_id':'figure','format':'png'}],
        'output_dir':str(out),'declared_outputs':[{'name':'report.pdf','format':'file'}]},outputs=[str(pdf)])
    rt=factory([producer],cid='pdf-dag');monkeypatch.setattr(registry,'get_config',lambda:rt.main.config);monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
    Image.new('RGB',(10,10),'blue').save(image)
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['make_pdf']['status'] in {'succeeded','failed'},timeout=45)
    state=rt.snapshot();assert state['nodes']['make_pdf']['status']=='succeeded',state['nodes']['make_pdf']
    owner={'username':'tester','conv_id':'pdf-dag'};entries=list_deliverables(root,state,owner)
    entry=next(x for x in entries if x['name']=='report.pdf')
    raw,kind,name=read_deliverable(root,state,owner,entry['artifact_id'],current_state=rt.snapshot)
    assert name=='report.pdf' and kind=='application/pdf' and raw==pdf.read_bytes()
    with pytest.raises(PermissionError):list_deliverables(root,state,{'username':'other','conv_id':'pdf-dag'})
    pdf.write_bytes(b'tampered')
    with pytest.raises(ValueError):read_deliverable(root,state,owner,entry['artifact_id'],current_state=rt.snapshot)
